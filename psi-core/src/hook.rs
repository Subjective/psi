//! The hook seam: compiled-in Rust hooks, registered at harness construction,
//! that run serially around every authoritative tool call. A before-hook
//! returns continue or block, and a block is reported to the model as a
//! refused call. Any future policy attaches here, including interactive
//! approval if it is ever wanted (docs/design.md, "Trusted environment and
//! hooks").

use crate::tool::{ToolInvocation, ToolOutput};

#[derive(Debug, Clone)]
pub enum HookDecision {
    Continue,
    Block { reason: String },
}

pub trait Hook: Send + Sync {
    fn before(&self, tool: &str, invocation: &ToolInvocation) -> HookDecision;

    /// Observes the output of a call that ran. Blocked and cancelled calls
    /// produce no tool output, so they reach no after-hook.
    fn after(&self, tool: &str, invocation: &ToolInvocation, output: &ToolOutput);
}

/// The registered hooks, run in registration order.
#[derive(Default)]
pub struct HookRegistry {
    hooks: Vec<Box<dyn Hook>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, hook: impl Hook + 'static) {
        self.hooks.push(Box::new(hook));
    }

    /// The first block wins; the hooks after it do not run.
    pub fn before(&self, tool: &str, invocation: &ToolInvocation) -> HookDecision {
        for hook in &self.hooks {
            if let HookDecision::Block { reason } = hook.before(tool, invocation) {
                return HookDecision::Block { reason };
            }
        }
        HookDecision::Continue
    }

    pub fn after(&self, tool: &str, invocation: &ToolInvocation, output: &ToolOutput) {
        for hook in &self.hooks {
            hook.after(tool, invocation, output);
        }
    }
}
