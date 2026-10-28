//! Vulkan queue management.
//!
//! This module provides the [`Queue`] type for submitting command buffers to the GPU.
//!
//! # Overview
//!
//! Queues schedule command buffers for GPU execution. Submissions to a queue are
//! guaranteed to **start** in order, but may **finish** out of order depending on
//! workload and GPU scheduling.
//!
//! For logically ordered sequences where each submission must complete before the
//! next begins, use [`Timeline`](crate::sync::Timeline) instead.

use ash::vk;

use crate::{
    Device, HasDevice,
    command::{CommandBuffer, CommandBufferState},
    fault::{DeviceResult, ResultWithDevice},
    utils::AsVkHandle,
};

/// A Vulkan command queue for scheduling GPU work.
///
/// Queues act as schedulers - submissions start in the order they are submitted,
/// but execution may complete out of order. For ordered execution guarantees,
/// use [`Timeline`](crate::sync::Timeline).
///
/// Each queue belongs to a queue family, which determines what operations it supports
/// (graphics, compute, transfer, etc.).
pub struct Queue {
    device: Device,
    handle: vk::Queue,
    family_index: u32,
    capabilities: vk::QueueFlags,

    /// Just a simple array to avoid repeated allocation during submission
    reused_semaphore_submit_info: Vec<vk::SemaphoreSubmitInfo<'static>>,
}

impl HasDevice for Queue {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl AsVkHandle for Queue {
    type Handle = vk::Queue;
    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}

impl Queue {
    /// Returns the queue family index.
    pub fn family_index(&self) -> u32 {
        self.family_index
    }
    /// Returns the capabilities supported by the queue family that this queue belongs to.
    pub fn capabilities(&self) -> vk::QueueFlags {
        self.capabilities
    }
    pub(crate) fn from_raw(
        device: Device,
        queue: vk::Queue,
        family_index: u32,
        caps: vk::QueueFlags,
    ) -> Self {
        Self {
            device,
            handle: queue,
            reused_semaphore_submit_info: Vec::new(),
            family_index,
            capabilities: caps,
        }
    }

    /// Submits a command buffer for execution.
    ///
    /// The command buffer must be in the `Executable` state (recording finished).
    /// After submission, it transitions to the `Pending` state.
    ///
    /// Note: Submission order does not guarantee completion order. Use
    /// [`Timeline`](crate::sync::Timeline) for sequential execution guarantees.
    pub fn submit(&mut self, cb: &mut CommandBuffer) -> DeviceResult<()> {
        // Safety: there are no semaphores besides the ones pumicite tracks.
        unsafe { self.submit_with_semaphores(cb, &[], &[]) }
    }

    /// Like [`submit`](Self::submit), but the submission also waits on each semaphore in
    /// `waits` and signals each semaphore in `signals`.
    ///
    /// # Safety
    /// - A binary semaphore in `waits` must have a pending signal that nothing else waits
    ///   on.
    /// - A binary semaphore in `signals` must be unsignaled, and nothing else may signal
    ///   it.
    /// - A timeline semaphore in `signals` must be signaled to a value greater than its
    ///   current value, and greater than any value that something else signals it to
    ///   before this submission does.
    /// - Every semaphore must stay alive until the submission finishes.
    pub unsafe fn submit_with_semaphores(
        &mut self,
        cb: &mut CommandBuffer,
        waits: &[vk::SemaphoreSubmitInfo<'static>],
        signals: &[vk::SemaphoreSubmitInfo<'static>],
    ) -> DeviceResult<()> {
        assert_eq!(
            cb.state(),
            CommandBufferState::Executable,
            "The command buffer must finish recording first!"
        );
        unsafe {
            assert!(self.reused_semaphore_submit_info.is_empty());
            let timestamp = cb
                .timestamp
                .as_ref()
                .expect("Command buffer must be scheduled on a timeline before submission");
            let waits = cb
                .wait_semaphores
                .iter()
                .map(|(semaphore, (value, stage_mask))| vk::SemaphoreSubmitInfo {
                    semaphore: semaphore.vk_handle(),
                    value: *value,
                    stage_mask: *stage_mask,
                    ..Default::default()
                })
                .chain(std::iter::once(vk::SemaphoreSubmitInfo {
                    semaphore: timestamp.semaphore().vk_handle(),
                    value: timestamp.value() - 1,
                    stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                    ..Default::default()
                }))
                .chain(waits.iter().cloned());
            self.reused_semaphore_submit_info.extend(waits);
            let wait_count = self.reused_semaphore_submit_info.len();
            let signals = std::iter::once(vk::SemaphoreSubmitInfo {
                semaphore: timestamp.semaphore().vk_handle(),
                stage_mask: vk::PipelineStageFlags2::ALL_COMMANDS,
                value: timestamp.value(),
                ..Default::default()
            })
            .chain(signals.iter().cloned());
            self.reused_semaphore_submit_info.extend(signals);
            let (waits, signals) = self.reused_semaphore_submit_info.split_at(wait_count);
            let result = self
                .device
                .queue_submit2(
                    self.handle,
                    &[vk::SubmitInfo2::default()
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo {
                            command_buffer: cb.buffer,
                            ..Default::default()
                        }])
                        .wait_semaphore_infos(waits)
                        .signal_semaphore_infos(signals)],
                    vk::Fence::null(),
                )
                .with_device(&self.device);
            self.reused_semaphore_submit_info.clear();
            result?;
            cb.wait_semaphores.clear();
            cb.state = CommandBufferState::Pending;
            Ok(())
        }
    }
}
