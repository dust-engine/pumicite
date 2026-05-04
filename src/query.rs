//! Query pool management.
//!
//! [`QueryPool`] wraps `VkQueryPool`. Queries within a pool are referenced by
//! index; before being written they must be reset via
//! [`QueryPool::host_reset`] or [`CommandEncoder::reset_query_pool`]. Once the
//! GPU has finished writing, results are read back with
//! [`QueryPool::get_results`].
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
    command::CommandEncoder,
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
    /// Requires Vulkan 1.2 or `VK_EXT_host_query_reset`. The range must not be
    /// in use by any submitted command buffer.
    pub fn host_reset(&self, range: Range<u32>) {
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
    pub fn reset_query_pool(&mut self, pool: &QueryPool, range: Range<u32>) {
        assert!(range.end <= pool.len, "query range out of bounds");
        unsafe {
            self.device().cmd_reset_query_pool(
                self.buffer().buffer,
                pool.handle,
                range.start,
                range.end - range.start,
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
        pool: &QueryPool,
        first_query: u32,
    ) {
        assert!(
            first_query + acceleration_structures.len() as u32 <= pool.len,
            "query range out of bounds",
        );
        unsafe {
            self.device()
                .extension::<ash::khr::acceleration_structure::Meta>()
                .cmd_write_acceleration_structures_properties(
                    self.buffer().buffer,
                    acceleration_structures,
                    pool.ty,
                    pool.handle,
                    first_query,
                );
        }
    }
}
