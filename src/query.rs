//! Query pool management.
//!
//! [`QueryPool`] wraps `VkQueryPool`. Queries within a pool are referenced by
//! index; before being written they must be reset via
//! [`QueryPool::host_reset`] or [`CommandEncoder::reset_query_pool`]. Once the
//! GPU has finished writing, results are read back with
//! [`QueryPool::get_results`].
//!
//! The safe methods take `&mut QueryPool` or a [`GPURefMut`] token, so whoever
//! resets or writes owns every query in the pool. To share one pool between the
//! host and several in-flight command buffers that each own different queries
//! (e.g. a ring of timestamp slots), use the `_unchecked` variants, which take a
//! shared reference and leave per-query ownership to the caller.
//!
//! # Example usage
//!
//! ```ignore
//! let pool = QueryPool::new(
//!     device.clone(),
//!     vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
//!     blases.len() as u32,
//! )?;
//!
//! cmd_pool.record(&mut cmd, |encoder| {
//!     encoder.reset_query_pool(&pool, 0..blases.len() as u32);
//!     encoder.write_acceleration_structures_properties(&blases, &pool, 0);
//! });
//! // ... submit, wait ...
//!
//! let mut sizes = vec![0u64; blases.len()];
//! pool.get_results(0, &mut sizes, vk::QueryResultFlags::TYPE_64)?;
//! ```

use std::{fmt::Debug, ops::Range};

use ash::{VkResult, vk};

use crate::{
    Device, HasDevice,
    command::{CommandEncoder, GPURef, GPURefMut, project_host_metadata},
    utils::AsVkHandle,
};

/// A pool of GPU queries.
///
/// `count` query slots of type `ty` are allocated up front. Slots are
/// referenced by 32-bit index in subsequent calls.
pub struct QueryPool {
    device: Device,
    handle: vk::QueryPool,
    ty: vk::QueryType,
    len: u32,
}
project_host_metadata! {
    impl[] QueryPool {
        fn len(&self) -> u32;
        fn ty(&self) -> vk::QueryType;
    }
}
impl Debug for QueryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.handle.fmt(f)
    }
}
impl HasDevice for QueryPool {
    fn device(&self) -> &Device {
        &self.device
    }
}
impl AsVkHandle for QueryPool {
    type Handle = vk::QueryPool;

    fn vk_handle(&self) -> Self::Handle {
        self.handle
    }
}

impl QueryPool {
    /// Creates a query pool with `count` queries of the given type.
    pub fn new(device: Device, ty: vk::QueryType, len: u32) -> VkResult<Self> {
        let handle = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo {
                    query_type: ty,
                    query_count: len,
                    ..Default::default()
                },
                None,
            )?
        };
        Ok(Self {
            device,
            handle,
            ty,
            len,
        })
    }

    /// Query type the pool was created with.
    pub fn ty(&self) -> vk::QueryType {
        self.ty
    }

    /// Total number of query slots in the pool.
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Resets queries in `range` from the host.
    ///
    /// Requires Vulkan 1.2 or `VK_EXT_host_query_reset`.
    pub fn host_reset(&mut self, range: Range<u32>) {
        // Safety: `&mut self` means no command buffer holds a token to the pool.
        unsafe { self.host_reset_unchecked(range) }
    }

    /// Resets queries in `range` from the host, through a shared reference.
    ///
    /// Requires Vulkan 1.2 or `VK_EXT_host_query_reset`.
    ///
    /// # Safety
    ///
    /// The caller must have exclusive ownership of the queries in `range`: every command
    /// buffer that reset or wrote them has completed, no command buffer that accesses them
    /// is submitted before this returns, and no other thread resets them concurrently.
    /// Other queries of the pool may be in use by the GPU.
    pub unsafe fn host_reset_unchecked(&self, range: Range<u32>) {
        assert!(range.end <= self.len, "query range out of bounds");
        unsafe {
            self.device
                .reset_query_pool(self.handle, range.start, range.end - range.start);
        }
    }

    /// Reads `data.len()` consecutive query results starting at `first_query`
    /// into `data`. Each query writes one `T` value.
    ///
    /// `flags` controls availability and wait behaviour. Pass
    /// [`vk::QueryResultFlags::WAIT`] to block until the GPU has finished
    /// writing, or omit it and handle [`vk::Result::NOT_READY`] explicitly.
    /// For 64-bit results (e.g. AS compacted sizes) include
    /// [`vk::QueryResultFlags::TYPE_64`].
    ///
    /// `size_of::<T>()` must equal the per-query result stride implied by the
    /// pool's query type and `flags` (i.e. 4 or 8 bytes per value, plus a
    /// trailing `u32`/`u64` if [`vk::QueryResultFlags::WITH_AVAILABILITY`] is
    /// set). Mismatched `T` produces well-defined but meaningless integers.
    pub fn get_results<T>(
        &self,
        first_query: u32,
        data: &mut [T],
        flags: vk::QueryResultFlags,
    ) -> VkResult<()> {
        assert!(
            first_query + data.len() as u32 <= self.len,
            "query range out of bounds",
        );
        unsafe {
            self.device
                .get_query_pool_results(self.handle, first_query, data, flags)
        }
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.handle, None);
        }
    }
}

impl<'a> CommandEncoder<'a> {
    /// Resets queries in `range` so they can be written. Every query must be
    /// reset before it is written to.
    pub fn reset_query_pool(
        &mut self,
        pool: impl Into<GPURefMut<'a, QueryPool>>,
        range: Range<u32>,
    ) {
        // Safety: the `GPURefMut` grants this command buffer every query in the pool.
        unsafe { self.reset_query_pool_unchecked(pool.into().readonly(), range) }
    }

    /// [`reset_query_pool`](Self::reset_query_pool) through a shared token.
    ///
    /// # Safety
    ///
    /// This command buffer must have exclusive ownership of the queries in `range` while it
    /// may execute: no other command buffer accesses them, and the host does not reset or
    /// read them, until it completes. Other queries of the pool may be in use elsewhere.
    pub unsafe fn reset_query_pool_unchecked(
        &mut self,
        pool: impl Into<GPURef<'a, QueryPool>>,
        range: Range<u32>,
    ) {
        let pool = pool.into();
        assert!(range.end <= pool.len(), "query range out of bounds");
        unsafe {
            self.device().cmd_reset_query_pool(
                self.buffer().buffer,
                pool.vk_handle(),
                range.start,
                range.end - range.start,
            );
        }
    }

    /// Records a GPU timestamp into slot `query` of `pool` once all previously
    /// submitted commands have reached `stage`.
    ///
    /// The pool must have been created with [`vk::QueryType::TIMESTAMP`], and
    /// the slot must have been reset (via [`CommandEncoder::reset_query_pool`])
    /// since it was last written. Timestamps may only be recorded on a queue
    /// whose family reports a non-zero `timestampValidBits`.
    ///
    /// Two timestamps bracketing a sequence of commands give the elapsed device
    /// time as `(end - start) * VkPhysicalDeviceLimits::timestampPeriod`
    /// nanoseconds, provided no counter overflow occurs.
    pub fn write_timestamp(
        &mut self,
        pool: impl Into<GPURefMut<'a, QueryPool>>,
        stage: vk::PipelineStageFlags2,
        query: u32,
    ) {
        // Safety: the `GPURefMut` grants this command buffer every query in the pool.
        unsafe { self.write_timestamp_unchecked(pool.into().readonly(), stage, query) }
    }

    /// [`write_timestamp`](Self::write_timestamp) through a shared token.
    ///
    /// # Safety
    ///
    /// This command buffer must have exclusive ownership of query `query` while it may
    /// execute: no other command buffer accesses it, and the host does not reset it, until
    /// it completes. The host may poll it with [`QueryPool::get_results`] and
    /// [`vk::QueryResultFlags::WITH_AVAILABILITY`]. Other queries of the pool may be in use
    /// elsewhere.
    pub unsafe fn write_timestamp_unchecked(
        &mut self,
        pool: impl Into<GPURef<'a, QueryPool>>,
        stage: vk::PipelineStageFlags2,
        query: u32,
    ) {
        let pool = pool.into();
        assert!(query < pool.len(), "query index out of bounds");
        debug_assert_eq!(
            pool.ty(),
            vk::QueryType::TIMESTAMP,
            "write_timestamp requires a TIMESTAMP query pool",
        );
        unsafe {
            self.device().cmd_write_timestamp2(
                self.buffer().buffer,
                stage,
                pool.vk_handle(),
                query,
            );
        }
    }

    /// Writes properties of `acceleration_structures` into consecutive query
    /// slots starting at `first_query`. The pool's query type selects which
    /// property is written (e.g.
    /// [`vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR`]).
    ///
    /// The acceleration structures must remain valid until the recorded
    /// command buffer completes execution; retain or lock them on the encoder
    /// as needed. Requires `VK_KHR_acceleration_structure`.
    pub fn write_acceleration_structures_properties(
        &mut self,
        acceleration_structures: &[vk::AccelerationStructureKHR],
        pool: impl Into<GPURefMut<'a, QueryPool>>,
        first_query: u32,
    ) {
        // Safety: the `GPURefMut` grants this command buffer every query in the pool.
        unsafe {
            self.write_acceleration_structures_properties_unchecked(
                acceleration_structures,
                pool.into().readonly(),
                first_query,
            )
        }
    }

    /// [`write_acceleration_structures_properties`](Self::write_acceleration_structures_properties)
    /// through a shared token.
    ///
    /// # Safety
    ///
    /// This command buffer must have exclusive ownership of the queries
    /// `first_query..first_query + acceleration_structures.len()` while it may execute: no
    /// other command buffer accesses them, and the host does not reset them, until it
    /// completes. Other queries of the pool may be in use elsewhere.
    pub unsafe fn write_acceleration_structures_properties_unchecked(
        &mut self,
        acceleration_structures: &[vk::AccelerationStructureKHR],
        pool: impl Into<GPURef<'a, QueryPool>>,
        first_query: u32,
    ) {
        let pool = pool.into();
        assert!(
            first_query + acceleration_structures.len() as u32 <= pool.len(),
            "query range out of bounds",
        );
        unsafe {
            self.device()
                .extension::<ash::khr::acceleration_structure::Meta>()
                .cmd_write_acceleration_structures_properties(
                    self.buffer().buffer,
                    acceleration_structures,
                    pool.ty(),
                    pool.vk_handle(),
                    first_query,
                );
        }
    }
}
