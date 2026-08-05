//! Artificial tool latency. A benchmark's tools do real work against a
//! fixture workspace, which takes microseconds; real tool calls do not. Every
//! tool is wrapped so its calls take the time the measured profile says they
//! should, which is what makes a baseline comparable to a real session and
//! what Milestone 6's speculation has to hide.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::tool::{Tool, ToolEffect, ToolFuture, ToolInvocation, ToolRegistry, ToolSpec};

/// The seed every latency stream starts from. Fixed rather than configurable:
/// repeated trials of one task must inject the same latencies, or their
/// measurements are not of the same run.
const SEED: u64 = 0x0f5e_ed00_0000_0001;

/// A tool's latency distribution, given as the two numbers real sessions were
/// measured by. Draws are lognormal fitted to them, which reproduces a heavy
/// tail from a median and a p95 alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyProfile {
    pub median_ms: u64,
    pub p95_ms: u64,
}

impl LatencyProfile {
    /// Every call takes the same time. A stream with spread is reproducible
    /// too, but only a fixed profile gives a test a latency it can name.
    pub const fn fixed(ms: u64) -> Self {
        Self {
            median_ms: ms,
            p95_ms: ms,
        }
    }
}

/// The latency the successive calls to one tool take. One stream per tool,
/// seeded from the tool's name, so the nth call to a tool takes the same time
/// in every trial however the calls to other tools interleave.
pub struct LatencyStream {
    profile: LatencyProfile,
    prng: Prng,
}

impl LatencyStream {
    pub fn new(tool: &str, profile: LatencyProfile) -> Self {
        Self {
            profile,
            prng: Prng::new(SEED ^ fnv1a(tool)),
        }
    }

    /// The next call's latency. A profile whose median is its p95 has no
    /// spread, so it returns that number every time.
    pub fn next_ms(&mut self) -> u64 {
        let LatencyProfile { median_ms, p95_ms } = self.profile;
        if p95_ms <= median_ms {
            return median_ms;
        }
        // 1.6449 is the standard normal's 95th percentile.
        let sigma = ((p95_ms as f64) / (median_ms as f64)).ln() / 1.6449;
        let drawn = (median_ms as f64) * (sigma * self.prng.next_normal()).exp();
        // The real tail runs much further than this — a REPL under `exec` can
        // block for a minute — but one draw must not dominate a whole run, so
        // it is cut at four times the p95.
        drawn.clamp(0.0, (p95_ms * 4) as f64).round() as u64
    }
}

/// Which profile each tool draws from.
#[derive(Debug, Clone)]
pub struct Latency {
    default: LatencyProfile,
    per_tool: Vec<(String, LatencyProfile)>,
}

impl Latency {
    /// The measured profile: across 670 real Codex sessions the active tool
    /// median is ~40ms with a p95 near 2s, and `exec` alone is slower in the
    /// middle and slightly lighter in the tail.
    pub fn measured() -> Self {
        Self {
            default: LatencyProfile {
                median_ms: 40,
                p95_ms: 2_000,
            },
            per_tool: Vec::new(),
        }
        .with_tool(
            "exec",
            LatencyProfile {
                median_ms: 70,
                p95_ms: 1_800,
            },
        )
    }

    /// Every tool takes the same fixed time.
    pub fn fixed(ms: u64) -> Self {
        Self {
            default: LatencyProfile::fixed(ms),
            per_tool: Vec::new(),
        }
    }

    /// Overrides one tool's profile.
    pub fn with_tool(mut self, tool: &str, profile: LatencyProfile) -> Self {
        self.per_tool.retain(|(name, _)| name != tool);
        self.per_tool.push((tool.to_string(), profile));
        self
    }

    pub fn profile(&self, tool: &str) -> LatencyProfile {
        self.per_tool
            .iter()
            .find(|(name, _)| name == tool)
            .map(|(_, profile)| *profile)
            .unwrap_or(self.default)
    }
}

/// A tool that sleeps before it runs. The engine measures a call around the
/// whole future, so the injected time lands in the `tool_result` item's
/// duration exactly as a slow tool's own time would.
pub struct SlowTool {
    inner: Arc<dyn Tool>,
    stream: Mutex<LatencyStream>,
}

impl Tool for SlowTool {
    fn spec(&self) -> ToolSpec {
        self.inner.spec()
    }

    fn effect(&self) -> ToolEffect {
        self.inner.effect()
    }

    fn execute(&self, invocation: ToolInvocation) -> ToolFuture {
        // Drawn here rather than inside the future, so the draw order is the
        // call order whatever the runtime does with the future.
        let delay = self.stream.lock().expect("latency lock").next_ms();
        let inner = self.inner.clone();
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(delay)).await;
            inner.execute(invocation).await
        })
    }
}

/// Wraps every tool of a profile in its injected latency, keeping the
/// advertised specs and declared effects untouched: the model and the engine
/// see the same tools, only slower.
pub fn inject_latency(registry: ToolRegistry, latency: &Latency) -> ToolRegistry {
    let mut slowed = ToolRegistry::new();
    for tool in registry.tools() {
        let name = tool.spec().name;
        slowed.register(SlowTool {
            inner: tool.clone(),
            stream: Mutex::new(LatencyStream::new(&name, latency.profile(&name))),
        });
    }
    slowed
}

/// xorshift64*. Hand-rolled because the only randomness Psi needs is a
/// reproducible latency draw, which is not worth a dependency.
struct Prng(u64);

impl Prng {
    fn new(seed: u64) -> Self {
        // Zero is xorshift's fixed point, so it can never be the state.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A uniform in (0, 1): 53 bits of mantissa, offset so neither endpoint
    /// can come out, because the transform below takes a logarithm.
    fn next_unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// A standard normal, by Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let (u1, u2) = (self.next_unit(), self.next_unit());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// FNV-1a, to turn a tool's name into its stream's seed.
fn fnv1a(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
