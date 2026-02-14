//! Callback manager for dispatching events to multiple callbacks

use super::traits::{CallbackAction, CallbackContext, TrainerCallback};

/// Manages multiple callbacks and dispatches events
pub struct CallbackManager {
    callbacks: Vec<Box<dyn TrainerCallback>>,
}

impl CallbackManager {
    /// Create new callback manager
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback
    pub fn add<C: TrainerCallback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }

    /// Check if no callbacks are registered
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }

    /// Get number of callbacks
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Fire train begin event
    pub fn on_train_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_train_begin(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire train end event
    pub fn on_train_end(&mut self, ctx: &CallbackContext) {
        for cb in &mut self.callbacks {
            cb.on_train_end(ctx);
        }
    }

    /// Fire epoch begin event
    pub fn on_epoch_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            match cb.on_epoch_begin(ctx) {
                CallbackAction::Stop => return CallbackAction::Stop,
                CallbackAction::SkipEpoch => return CallbackAction::SkipEpoch,
                CallbackAction::Continue => {}
            }
        }
        CallbackAction::Continue
    }

    /// Fire epoch end event
    pub fn on_epoch_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_epoch_end(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire step begin event
    pub fn on_step_begin(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_step_begin(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }

    /// Fire step end event
    pub fn on_step_end(&mut self, ctx: &CallbackContext) -> CallbackAction {
        for cb in &mut self.callbacks {
            if cb.on_step_end(ctx) == CallbackAction::Stop {
                return CallbackAction::Stop;
            }
        }
        CallbackAction::Continue
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::callback::{EarlyStopping, ProgressCallback};

    #[test]
    fn test_callback_manager_dispatch() {
        let mut manager = CallbackManager::new();

        // Add early stopping that triggers after 1 epoch without improvement
        let es = EarlyStopping::new(1, 0.001);
        manager.add(es);

        let mut ctx = CallbackContext::default();
        ctx.loss = 1.0;

        // First epoch
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);

        // Second epoch - no improvement, should stop
        ctx.epoch = 1;
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_len_and_empty() {
        let mut manager = CallbackManager::new();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);

        manager.add(ProgressCallback::new(10));
        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_callback_manager_on_train_begin_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(
            manager.on_train_begin(&CallbackContext::default()),
            CallbackAction::Stop
        );
    }

    #[test]
    fn test_callback_manager_on_train_end() {
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

        struct EndCallback {
            called: Arc<AtomicBool>,
        }
        impl TrainerCallback for EndCallback {
            fn on_train_end(&mut self, _: &CallbackContext) {
                self.called.store(true, Ordering::SeqCst);
            }
            fn name(&self) -> &'static str {
                "EndCallback"
            }
        }

        let called = Arc::new(AtomicBool::new(false));
        let mut manager = CallbackManager::new();
        manager.add(EndCallback {
            called: called.clone(),
        });
        manager.on_train_end(&CallbackContext::default());
        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_callback_manager_on_epoch_begin_skip() {
        struct SkipCallback;
        impl TrainerCallback for SkipCallback {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::SkipEpoch
            }
            fn name(&self) -> &'static str {
                "SkipCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(SkipCallback);
        assert_eq!(
            manager.on_epoch_begin(&CallbackContext::default()),
            CallbackAction::SkipEpoch
        );
    }

    #[test]
    fn test_callback_manager_on_epoch_begin_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(
            manager.on_epoch_begin(&CallbackContext::default()),
            CallbackAction::Stop
        );
    }

    #[test]
    fn test_callback_manager_on_step_begin_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(
            manager.on_step_begin(&CallbackContext::default()),
            CallbackAction::Stop
        );
    }

    #[test]
    fn test_callback_manager_default() {
        let manager = CallbackManager::default();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_callback_manager_stop_propagation() {
        // Create a callback that always returns Stop
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        manager.add(ProgressCallback::new(10));

        let ctx = CallbackContext::default();
        let action = manager.on_epoch_end(&ctx);
        // Stop should propagate
        assert_eq!(action, CallbackAction::Stop);
    }

    #[test]
    fn test_callback_manager_on_step_end_stop() {
        struct StopCallback;
        impl TrainerCallback for StopCallback {
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "StopCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(StopCallback);
        assert_eq!(
            manager.on_step_end(&CallbackContext::default()),
            CallbackAction::Stop
        );
    }

    #[test]
    fn test_callback_manager_all_continue() {
        // Test that all callbacks continue properly
        struct ContinueCallback;
        impl TrainerCallback for ContinueCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction {
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "ContinueCallback"
            }
        }

        let mut manager = CallbackManager::new();
        manager.add(ContinueCallback);
        manager.add(ContinueCallback);

        let ctx = CallbackContext::default();
        assert_eq!(manager.on_train_begin(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_epoch_begin(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_step_begin(&ctx), CallbackAction::Continue);
        assert_eq!(manager.on_step_end(&ctx), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_manager_multiple_train_end() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountingEndCallback {
            count: Arc<AtomicUsize>,
        }

        impl TrainerCallback for CountingEndCallback {
            fn on_train_end(&mut self, _: &CallbackContext) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            fn name(&self) -> &'static str {
                "CountingEndCallback"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(CountingEndCallback {
            count: count.clone(),
        });
        manager.add(CountingEndCallback {
            count: count.clone(),
        });
        manager.add(CountingEndCallback {
            count: count.clone(),
        });

        manager.on_train_end(&CallbackContext::default());
        assert_eq!(count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_callback_manager_stop_after_first() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        struct CountingStopCallback {
            count: Arc<AtomicUsize>,
        }

        impl TrainerCallback for CountingStopCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Stop
            }
            fn name(&self) -> &'static str {
                "CountingStopCallback"
            }
        }

        struct CountingContinueCallback {
            count: Arc<AtomicUsize>,
        }

        impl TrainerCallback for CountingContinueCallback {
            fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                self.count.fetch_add(1, Ordering::SeqCst);
                CallbackAction::Continue
            }
            fn name(&self) -> &'static str {
                "CountingContinueCallback"
            }
        }

        let count = Arc::new(AtomicUsize::new(0));
        let mut manager = CallbackManager::new();
        manager.add(CountingStopCallback {
            count: count.clone(),
        });
        manager.add(CountingContinueCallback {
            count: count.clone(),
        });

        // First callback stops, second should not be called
        let action = manager.on_train_begin(&CallbackContext::default());
        assert_eq!(action, CallbackAction::Stop);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::train::callback::EarlyStopping;
    use proptest::prelude::*;

    proptest! {
        /// Callback manager should propagate stop action
        #[test]
        fn callback_manager_propagates_stop(
            patience in 1usize..5,
        ) {
            let mut manager = CallbackManager::new();
            manager.add(EarlyStopping::new(patience, 0.001));

            let mut ctx = CallbackContext::default();
            ctx.loss = 1.0;

            // Should continue until patience exhausted
            for epoch in 0..patience {
                ctx.epoch = epoch;
                let action = manager.on_epoch_end(&ctx);
                if epoch < patience - 1 {
                    prop_assert_eq!(action, CallbackAction::Continue);
                }
            }

            // Final epoch should stop
            ctx.epoch = patience;
            prop_assert_eq!(manager.on_epoch_end(&ctx), CallbackAction::Stop);
        }

        /// Multiple callbacks should all fire
        #[test]
        fn multiple_callbacks_fire(
            num_callbacks in 1usize..5,
        ) {
            use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

            struct CounterCallback {
                counter: Arc<AtomicUsize>,
            }

            impl TrainerCallback for CounterCallback {
                fn on_train_begin(&mut self, _: &CallbackContext) -> CallbackAction {
                    self.counter.fetch_add(1, Ordering::SeqCst);
                    CallbackAction::Continue
                }
                fn on_train_end(&mut self, _: &CallbackContext) {}
                fn on_epoch_begin(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_epoch_end(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_step_begin(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn on_step_end(&mut self, _: &CallbackContext) -> CallbackAction { CallbackAction::Continue }
                fn name(&self) -> &'static str { "CounterCallback" }
            }

            let counter = Arc::new(AtomicUsize::new(0));
            let mut manager = CallbackManager::new();

            for _ in 0..num_callbacks {
                manager.add(CounterCallback { counter: counter.clone() });
            }

            let ctx = CallbackContext::default();
            manager.on_train_begin(&ctx);

            prop_assert_eq!(counter.load(Ordering::SeqCst), num_callbacks);
        }
    }
}
