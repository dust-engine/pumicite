//! Vulkan buffer abstractions with automatic memory management.
//!
//! This module provides safe wrappers around Vulkan buffers with different memory
//! allocation strategies optimized for various GPU architectures and use cases.
//!
//! # Choosing an Allocation Strategy
//!
//! - **[`Buffer::new_private`]**: GPU-exclusive memory. Use for render targets, scratch
//!   buffers, and any data generated entirely on the GPU.
//!
//! - **[`Buffer::new_upload`]**: Device-local memory that may be directly writable.
//!   Avoids staging copies on GPUs with resizable bars or integrated GPUs. Always use
//!   [`BufferExt::update_contents`] which handles both paths transparently.
//!
//! - **[`Buffer::new_host`]**: CPU-accessible memory in system RAM. Use for staging
//!   buffers or data that the GPU reads infrequently.
//!
//! - **[`Buffer::new_dynamic`]**: Host-cached system memory for CPU readback. Guarantees
//!   fast CPU reads. GPU writes can be slow on non-integrated GPUs.
//!
//! - **[`RingBuffer`]**: Suballocator for transient per-frame data. Ideal when you
//!   write data once per frame and don't need it to persist.
//!
//! - **[`ManagedBuffer`]**: Hybrid buffer that auto-selects between direct writes
//!   (integrated) and staging transfers (discrete). Good default for infrequently
//!   updated data that needs both fast CPU writes and fast GPU reads.

use std::{
    collections::VecDeque,
    ffi::{CStr, CString},
    fmt::Debug,
    ops::RangeBounds,
    sync::Arc,
};

use ash::{
    VkResult,
    vk::{self, TaggedStructure},
};
use vk_mem::Alloc;

use crate::{Allocator, Device, HasDevice, command::CommandEncoder, utils::AsVkHandle};

/// Common interface for Vulkan buffer types.
///
/// This trait abstracts over different buffer implementations ([`Buffer`],
/// [`RingBufferSuballocation`], [`ManagedBuffer`]) providing a unified interface
/// for accessing buffer properties and memory-mapped data.
///
/// # Memory Access
///
/// Buffers may or may not be host-visible depending on their allocation strategy.
/// Use [`as_slice`](BufferLike::as_slice) and [`as_slice_mut`](BufferLike::as_slice_mut)
/// to access mapped memory when available.
///
/// For non-coherent memory, call [`flush`](BufferLike::flush) after writes and
/// [`invalidate`](BufferLike::invalidate) before reads to ensure visibility.
pub trait BufferLike: AsVkHandle<Handle = vk::Buffer> + Send + Sync + 'static {
    /// Returns the offset within the underlying buffer.
    ///
    /// For standalone buffers this is always 0. For suballocations (like
    /// [`RingBufferSuballocation`]), this returns the offset within the parent buffer.
    fn offset(&self) -> vk::DeviceSize;

    /// Returns the buffer device address for use in shaders.
    ///
    /// Returns 0 if the buffer was not created with `SHADER_DEVICE_ADDRESS` usage.
    fn device_address(&self) -> vk::DeviceAddress;

    /// Returns the size of the buffer in bytes.
    fn size(&self) -> vk::DeviceSize;

    /// Returns a read-only slice of the buffer's mapped memory, if host-visible.
    ///
    /// Returns `None` if the buffer is not host-visible. Logs a warning if reading
    /// from memory that is not `HOST_CACHED`, as this may be slow.
    fn as_slice(&self) -> Option<&[u8]>;

    /// Returns a mutable slice of the buffer's mapped memory, if host-visible.
    ///
    /// Returns `None` if the buffer is not host-visible or not mapped.
    fn as_slice_mut(&mut self) -> Option<&mut [u8]>;

    /// Flushes the specified range to make CPU writes visible to the GPU.
    ///
    /// This is a no-op for `HOST_COHERENT` memory.
    fn flush(&mut self, range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()>;

    /// Invalidates the specified range to make GPU writes visible to the CPU.
    ///
    /// This is a no-op for `HOST_COHERENT` memory.
    fn invalidate(&mut self, range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()>;
}

/// A buffer fully bound to a memory allocation.
///
/// Buffer types:
///
/// |         | Discrete      | ReBar         | AMD Integrated | Intel/Apple Integrated |
/// |---------|---------------|--------------|-------------|----------------|
/// | Host    |   HOST_VISIBLE       |   HOST_VISIBLE    |   HOST_VISIBLE   | HOST_VISIBLE |
/// | Private |   DEVICE_LOCAL       |   DEVICE_LOCAL    |   DEVICE_LOCAL   |  DEVICE_LOCAL   |
/// | Dynamic |   HOST_VISBLE, HOST_CACHED   |  HOST_VISBLE, HOST_CACHED      |  HOST_VISBLE, HOST_CACHED     | HOST_VISIBLE, HOST_CACHED, DEVICE_LOCAL |
/// | Upload  |   DEVICE_LOCAL      |  DEVICE_LOCAL, HOST_VISIBLE   |  HOST_VISBLE   | HOST_VISIBLE, DEVICE_LOCAL |
pub struct Buffer {
    allocator: Allocator,
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    size: vk::DeviceSize,
    device_address: vk::DeviceAddress,

    memory_properties: vk::MemoryPropertyFlags,
}
impl HasDevice for Buffer {
    fn device(&self) -> &crate::Device {
        self.allocator.device()
    }
}
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}
impl Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("size", &self.size)
            .field("device_address", &self.device_address)
            .field("memory_properties", &self.memory_properties)
            .finish_non_exhaustive()
    }
}
impl crate::utils::AsVkHandle for Buffer {
    fn vk_handle(&self) -> Self::Handle {
        self.buffer
    }
    type Handle = vk::Buffer;
}
impl BufferLike for Buffer {
    fn offset(&self) -> vk::DeviceSize {
        0
    }

    fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    fn size(&self) -> vk::DeviceSize {
        self.size
    }
    fn as_slice(&self) -> Option<&[u8]> {
        if !self
            .memory_properties
            .contains(vk::MemoryPropertyFlags::HOST_CACHED)
        {
            tracing::warn!("Trying to read from buffer that isn't HOST_CACHED");
        }
        if self
            .memory_properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            Some(unsafe {
                std::slice::from_raw_parts(
                    self.allocator
                        .get_allocation_info(&self.allocation)
                        .mapped_data as *const u8,
                    self.size as usize,
                )
            })
        } else {
            None
        }
    }
    fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        if self
            .memory_properties
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        {
            unsafe {
                let mapped_data = self
                    .allocator
                    .get_allocation_info(&self.allocation)
                    .mapped_data as *mut u8;
                if mapped_data.is_null() {
                    None
                } else {
                    Some(std::slice::from_raw_parts_mut(
                        mapped_data,
                        self.size as usize,
                    ))
                }
            }
        } else {
            None
        }
    }

    fn flush(&mut self, range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        if self
            .memory_properties
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(());
        }
        let offset = match range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(end) => end + 1,
            std::ops::Bound::Excluded(end) => *end,
            std::ops::Bound::Unbounded => self.size,
        };
        self.allocator()
            .flush_allocation(&self.allocation, offset, end - offset)
    }

    fn invalidate(&mut self, range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        if self
            .memory_properties
            .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            return Ok(());
        }
        let offset = match range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start + 1,
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Included(end) => end + 1,
            std::ops::Bound::Excluded(end) => *end,
            std::ops::Bound::Unbounded => self.size,
        };
        self.allocator()
            .invalidate_allocation(&self.allocation, offset, end - offset)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.allocator
                .destroy_buffer(self.buffer, &mut self.allocation);
        }
    }
}

impl Buffer {
    pub fn allocator(&self) -> &Allocator {
        &self.allocator
    }
    pub fn from_raw(
        allocator: Allocator,
        buffer: vk::Buffer,
        allocation: vk_mem::Allocation,
        usage: vk::BufferUsageFlags,
        size: vk::DeviceSize,
    ) -> Self {
        let info = allocator.get_allocation_info(&allocation);
        let device_address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            unsafe {
                allocator
                    .device()
                    .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                        buffer,
                        ..Default::default()
                    })
            }
        } else {
            0
        };

        Self {
            memory_properties: allocator
                .device()
                .physical_device()
                .properties()
                .memory_types()[info.memory_type as usize]
                .property_flags,
            allocator,
            buffer,
            allocation,
            size,
            device_address,
        }
    }
    /// Create a buffer that is accessible exclusively from the GPU.
    ///
    /// Use for render targets, scratch buffers, and any data generated entirely on the GPU.
    ///
    /// Uses the pre-calculated `private` memory type from [`MemoryTypeMap`](crate::physical_device::MemoryTypeMap).
    pub fn new_private(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        let memory_type = allocator
            .device()
            .physical_device()
            .properties()
            .memory_type_map()
            .private;
        unsafe {
            let (buffer, allocation) = allocator.create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type,
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    flags: vk_mem::AllocationCreateFlags::empty(),
                    ..Default::default()
                },
                alignment,
            )?;
            Ok(Self::from_raw(allocator, buffer, allocation, usage, size))
        }
    }

    /// Create a HOST_VISIBLE buffer.
    ///
    /// Use for staging buffers or data that the GPU reads infrequently.
    ///
    /// Uses the pre-calculated `host` memory type from [`MemoryTypeMap`](crate::physical_device::MemoryTypeMap).
    pub fn new_host(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        let memory_type = allocator
            .device()
            .physical_device()
            .properties()
            .memory_type_map()
            .host;
        unsafe {
            let (buffer, allocation) = allocator.create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type,
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                alignment,
            )?;
            Ok(Self::from_raw(allocator, buffer, allocation, usage, size))
        }
    }

    /// Create a DEVICE_LOCAL buffer that is **preferably** host-writable.
    ///
    /// On GPUs with resizable BAR or integrated GPUs, the buffer is host-visible and
    /// uploads can be done directly. On discrete GPUs without resizable BAR, a staging
    /// buffer is required.
    ///
    /// Call [`BufferExt::update_contents`] to update its content transparently.
    ///
    /// Uses the pre-calculated `upload` memory type from [`MemoryTypeMap`](crate::physical_device::MemoryTypeMap).
    pub fn new_upload(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        mut usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        let memory_type_map = allocator
            .device()
            .physical_device()
            .properties()
            .memory_type_map();

        if !memory_type_map.upload_host_visible {
            usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }

        unsafe {
            let (buffer, allocation) = allocator.create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type_map.upload,
                    usage: vk_mem::MemoryUsage::AutoPreferDevice,
                    flags: vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_ALLOW_TRANSFER_INSTEAD,
                    ..Default::default()
                },
                alignment,
            )?;
            Ok(Self::from_raw(allocator, buffer, allocation, usage, size))
        }
    }

    /// Create a host-visible buffer that is **always** host-cached and **preferably** device-local.
    ///
    /// Useful for CPU readback. Guarantees fast CPU reads. GPU writes can be slow on
    /// non-integrated GPUs.
    ///
    /// Uses the pre-calculated `dynamic` memory type from [`MemoryTypeMap`](crate::physical_device::MemoryTypeMap).
    pub fn new_dynamic(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        let memory_type = allocator
            .device()
            .physical_device()
            .properties()
            .memory_type_map()
            .dynamic;
        unsafe {
            let (buffer, allocation) = allocator.create_buffer_with_alignment(
                &vk::BufferCreateInfo {
                    size,
                    usage,
                    ..Default::default()
                },
                &vk_mem::AllocationCreateInfo {
                    memory_type_bits: 1 << memory_type,
                    usage: vk_mem::MemoryUsage::AutoPreferHost,
                    flags: vk_mem::AllocationCreateFlags::MAPPED
                        | vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM,
                    ..Default::default()
                },
                alignment,
            )?;
            Ok(Self::from_raw(allocator, buffer, allocation, usage, size))
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.allocator
            .get_allocation_info(&self.allocation)
            .mapped_data as *const u8
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.allocator
            .get_allocation_info(&self.allocation)
            .mapped_data as *mut u8
    }
}

/// A single memory chunk used by [`RingBuffer`].
///
/// Each chunk is a contiguous Vulkan buffer with an dedicated allocation. The ring buffer manages
/// multiple chunks, recycling them when they are no longer in use by the GPU.
struct RingBufferChunk {
    device: Device,
    device_buffer: vk::Buffer,
    device_memory: vk::DeviceMemory,
    device_address: vk::DeviceAddress,
    ptr: *mut u8,
    size: u64,
}
impl Drop for RingBufferChunk {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.device_buffer, None);
            self.device.free_memory(self.device_memory, None);
        }
    }
}
unsafe impl Send for RingBufferChunk {}
unsafe impl Sync for RingBufferChunk {}
impl RingBufferChunk {
    fn new(
        device: Device,
        size: u64,
        memory_type_index: u32,
        buffer_usage: vk::BufferUsageFlags,
        memory_alloc_flags: vk::MemoryAllocateFlags,
        debug_name: &CStr,
    ) -> Self {
        unsafe {
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo {
                        usage: buffer_usage,
                        size,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            device.set_debug_name(buffer, debug_name).ok();

            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: size,
                        memory_type_index,
                        ..Default::default()
                    }
                    .push(&mut vk::MemoryAllocateFlagsInfo {
                        flags: memory_alloc_flags,
                        ..Default::default()
                    }),
                    None,
                )
                .unwrap();
            device.set_debug_name(memory, debug_name).ok();
            device.bind_buffer_memory(buffer, memory, 0).unwrap();
            let is_host_visible = device.physical_device().properties().memory_types()
                [memory_type_index as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
            let ptr = if is_host_visible {
                device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .unwrap() as *mut u8
            } else {
                std::ptr::null_mut()
            };
            let device_address =
                if buffer_usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                    device.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                        buffer,
                        ..Default::default()
                    })
                } else {
                    0
                };
            Self {
                device,
                device_buffer: buffer,
                device_memory: memory,
                device_address,
                ptr,
                size,
            }
        }
    }
}

/// A ring buffer allocator for transient GPU data.
///
/// Ring buffers are ideal for per-frame data that is written once by the CPU
/// and read once by the GPU (e.g., uniform data, draw parameters). The allocator
/// manages multiple chunks, allocating from the current chunk and recycling old
/// chunks once they are no longer in use.
///
/// # Chunk Recycling
///
/// Chunks are tracked via `Arc` reference counting. When the strong count drops
/// to 1 (only the ring buffer holds a reference), the chunk is considered free
/// and will be reused for new allocations.
pub struct RingBuffer {
    device: Device,
    chunk_size: vk::DeviceSize,
    current_chunk_head: u64,
    current_chunk: Option<Arc<RingBufferChunk>>,
    used_chunks: VecDeque<Arc<RingBufferChunk>>,
    memory_type_index: u32,
    buffer_usage: vk::BufferUsageFlags,
    memory_alloc_flags: vk::MemoryAllocateFlags,
    debug_name: &'static str,
}
impl HasDevice for RingBuffer {
    fn device(&self) -> &Device {
        &self.device
    }
}

impl RingBuffer {
    pub fn new(
        device: Device,
        chunk_size: vk::DeviceSize,
        memory_type_index: u32,
        buffer_usage: vk::BufferUsageFlags,
        memory_alloc_flags: vk::MemoryAllocateFlags,
        debug_name: &'static str,
    ) -> Self {
        RingBuffer {
            device,
            chunk_size,
            current_chunk_head: 0,
            current_chunk: None,
            used_chunks: VecDeque::new(),
            memory_type_index,
            buffer_usage,
            memory_alloc_flags,
            debug_name,
        }
    }

    pub fn allocate_buffer(
        &mut self,
        size: vk::DeviceSize,
        alignment: u64,
    ) -> RingBufferSuballocation {
        if size > self.chunk_size {
            // Double the chunk size.
            unimplemented!()
        }
        let aligned_start = self.current_chunk_head.next_multiple_of(alignment);
        let end = aligned_start + size;

        if let Some(current_chunk) = self.current_chunk.as_mut()
            && end <= current_chunk.size
        {
            // there's enough space
            self.current_chunk_head = end;
            return RingBufferSuballocation {
                buffer: current_chunk.device_buffer,
                offset: aligned_start,
                size,
                device_address: current_chunk.device_address + aligned_start,
                ptr: if current_chunk.ptr.is_null() {
                    std::ptr::null_mut()
                } else {
                    unsafe { current_chunk.ptr.add(aligned_start as usize) }
                },
                _chunk: current_chunk.clone(),
            };
        }
        // not enough space at the back of the belt.
        // try reuse heads of the belt.
        if let Some(peek) = self.used_chunks.front()
            && Arc::strong_count(peek) == 1
        {
            // No one else is using this chunk. Reuse this chunk.
            let reused_chunk = self.used_chunks.pop_front().unwrap();
            self.current_chunk_head = 0;

            if let Some(full_chunk) = self.current_chunk.take() {
                // Put the chunk at the head of the queue
                self.used_chunks.push_back(full_chunk);
            }
            self.current_chunk = Some(reused_chunk);
        } else {
            if let Some(full_chunk) = self.current_chunk.take() {
                // Put the chunk at the head of the queue
                self.used_chunks.push_back(full_chunk);
            }

            // Can't reuse any old chunks, so we need to allocate a new one
            let new_chunk = RingBufferChunk::new(
                self.device.clone(),
                self.chunk_size,
                self.memory_type_index,
                self.buffer_usage,
                self.memory_alloc_flags,
                CString::new(format!(
                    "{} chunk {}",
                    self.debug_name,
                    self.used_chunks.len()
                ))
                .unwrap()
                .as_c_str(),
            );
            tracing::info!(
                "Allocating new ring buffer chunk with size = {} for {}",
                self.chunk_size,
                self.debug_name
            );
            self.current_chunk_head = 0;
            self.current_chunk = Some(Arc::new(new_chunk));
        }

        // Now, retry the allocation
        let chunk = self.current_chunk.as_ref().unwrap().clone();

        let suballocation = RingBufferSuballocation {
            buffer: chunk.device_buffer,
            offset: 0,
            size,
            ptr: chunk.ptr,
            device_address: chunk.device_address,
            _chunk: chunk,
        };
        self.current_chunk_head += size;
        suballocation
    }
}

/// A suballocation from a [`RingBuffer`].
///
/// This represents a portion of a ring buffer chunk. The suballocation holds
/// an `Arc` reference to its parent chunk, keeping the chunk alive until
/// all suballocations from it are dropped.
///
/// Implements [`BufferLike`] for uniform access to buffer properties.
#[derive(Clone)]
pub struct RingBufferSuballocation {
    buffer: vk::Buffer,
    // The start of the suballocation block, including the alignment padding
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    ptr: *mut u8,
    device_address: vk::DeviceAddress,
    _chunk: Arc<RingBufferChunk>,
}
unsafe impl Send for RingBufferSuballocation {}
unsafe impl Sync for RingBufferSuballocation {}
impl AsVkHandle for RingBufferSuballocation {
    type Handle = vk::Buffer;
    fn vk_handle(&self) -> Self::Handle {
        self.buffer
    }
}
impl BufferLike for RingBufferSuballocation {
    fn offset(&self) -> vk::DeviceSize {
        self.offset
    }
    fn size(&self) -> vk::DeviceSize {
        self.size
    }
    fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }
    fn as_slice(&self) -> Option<&[u8]> {
        if self.ptr.is_null() {
            None
        } else {
            unsafe { Some(std::slice::from_raw_parts(self.ptr, self.size as usize)) }
        }
    }
    fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        if self.ptr.is_null() {
            None
        } else {
            unsafe { Some(std::slice::from_raw_parts_mut(self.ptr, self.size as usize)) }
        }
    }
    fn flush(&mut self, _range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        // TODO
        Ok(())
    }
    fn invalidate(&mut self, _range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        // TODO
        Ok(())
    }
}

/// A buffer that abstracts over the differences between integrated and discrete GPUs.
/// On integrated GPUs, memory is unified so a single buffer
/// serves both CPU and GPU access. On discrete GPUs, this uses a host buffer for CPU
/// writes and a device buffer for GPU access, with explicit transfers between them.
///
/// Can be treated as DEVICE_LOCAL, HOST_VISBLE, HOST_CACHED, but non-coherent memory.
///
/// # Usage
///
/// 1. Write data via [`as_slice_mut`](ManagedBuffer::as_slice_mut)
/// 2. Call [`flush`](ManagedBuffer::flush) with a command encoder to transfer to the GPU
/// 3. GPU reads from the buffer's device address
///
/// # Variants
///
/// - `Transfer`: Separate host and device buffers (discrete GPUs)
/// - `Direct`: Single unified buffer (integrated GPUs)
pub enum ManagedBuffer {
    Transfer {
        host: Arc<Buffer>,
        device: Arc<Buffer>,
    },
    Direct {
        buffer: Arc<Buffer>,
    },
}

impl ManagedBuffer {
    pub fn new(
        allocator: Allocator,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> VkResult<Self> {
        if allocator
            .device()
            .physical_device()
            .properties()
            .device_type
            == vk::PhysicalDeviceType::INTEGRATED_GPU
        {
            let buffer = Buffer::new_dynamic(allocator, size, alignment, usage)?;
            Ok(Self::Direct {
                buffer: Arc::new(buffer),
            })
        } else {
            let host = Buffer::new_dynamic(
                allocator.clone(),
                size,
                alignment,
                vk::BufferUsageFlags::TRANSFER_SRC,
            )?;
            let device = Buffer::new_private(
                allocator,
                size,
                alignment,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
            )?;
            Ok(Self::Transfer {
                host: Arc::new(host),
                device: Arc::new(device),
            })
        }
    }
    pub fn as_slice(&self) -> &[u8] {
        match self {
            ManagedBuffer::Transfer { host, .. } => host.as_slice().unwrap(),
            ManagedBuffer::Direct { buffer } => buffer.as_slice().unwrap(),
        }
    }
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        match self {
            ManagedBuffer::Transfer { host, .. } => unsafe {
                std::slice::from_raw_parts_mut(host.as_ptr() as *mut u8, host.size() as usize)
            },
            ManagedBuffer::Direct { buffer } => unsafe {
                std::slice::from_raw_parts_mut(buffer.as_ptr() as *mut u8, buffer.size() as usize)
            },
        }
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            ManagedBuffer::Transfer { host, .. } => host.as_ptr() as *mut u8,
            ManagedBuffer::Direct { buffer } => buffer.as_ptr() as *mut u8,
        }
    }
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            ManagedBuffer::Transfer { host, .. } => host.as_ptr(),
            ManagedBuffer::Direct { buffer } => buffer.as_ptr(),
        }
    }
    pub fn allocator(&self) -> &Allocator {
        match self {
            ManagedBuffer::Transfer { host, .. } => host.allocator(),
            ManagedBuffer::Direct { buffer } => buffer.allocator(),
        }
    }
    pub fn size(&self) -> u64 {
        match self {
            ManagedBuffer::Transfer { device, .. } => device.size(),
            ManagedBuffer::Direct { buffer } => buffer.size(),
        }
    }
    pub fn flush(&self, encoder: &mut CommandEncoder<'_>) {
        match self {
            ManagedBuffer::Transfer { host, device } => {
                let host = encoder.retain(host.clone());
                let device = encoder.retain(device.clone());
                encoder.copy_buffer(host.as_ref(), device.as_ref());
            }
            ManagedBuffer::Direct { .. } => {}
        }
    }
}
impl AsVkHandle for ManagedBuffer {
    type Handle = vk::Buffer;

    fn vk_handle(&self) -> Self::Handle {
        match self {
            ManagedBuffer::Transfer { device, .. } => device.vk_handle(),
            ManagedBuffer::Direct { buffer } => buffer.vk_handle(),
        }
    }
}
impl BufferLike for ManagedBuffer {
    fn offset(&self) -> vk::DeviceSize {
        0
    }

    fn device_address(&self) -> vk::DeviceAddress {
        match self {
            ManagedBuffer::Transfer { device, .. } => device.device_address(),
            ManagedBuffer::Direct { buffer } => buffer.device_address(),
        }
    }

    fn size(&self) -> vk::DeviceSize {
        match self {
            ManagedBuffer::Transfer { device, .. } => device.size,
            ManagedBuffer::Direct { buffer } => buffer.size,
        }
    }

    fn as_slice(&self) -> Option<&[u8]> {
        Some(ManagedBuffer::as_slice(self))
    }

    fn as_slice_mut(&mut self) -> Option<&mut [u8]> {
        Some(ManagedBuffer::as_slice_mut(self))
    }

    fn flush(&mut self, _range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        // TODO
        Ok(())
    }

    fn invalidate(&mut self, _range: impl RangeBounds<vk::DeviceSize>) -> VkResult<()> {
        // TODO
        Ok(())
    }
}

/// Trait for types that can allocate staging buffers.
///
/// Staging buffers are temporary host-visible buffers used to transfer data
/// to device-local memory. This trait is implemented by [`RingBuffer`] (for
/// efficient transient allocations) and [`Allocator`] (for standalone buffers).
///
/// Used by [`BufferExt::update_contents`] and [`ImageExt::update_contents_async`](crate::image::ImageExt::update_contents_async).
pub trait StagingBufferAllocator {
    /// The buffer type returned by this allocator.
    type Buffer: BufferLike;

    /// Allocates a staging buffer of at least the specified size.
    ///
    /// The returned buffer must be host-visible for CPU writes.
    fn allocate_staging_buffer(&mut self, size: u64) -> VkResult<Self::Buffer>;
}

impl StagingBufferAllocator for RingBuffer {
    type Buffer = RingBufferSuballocation;
    fn allocate_staging_buffer(&mut self, size: u64) -> VkResult<RingBufferSuballocation> {
        Ok(self.allocate_buffer(size, 4))
    }
}
impl StagingBufferAllocator for Allocator {
    type Buffer = Buffer;
    fn allocate_staging_buffer(&mut self, size: u64) -> VkResult<Buffer> {
        Buffer::new_host(self.clone(), size, 4, vk::BufferUsageFlags::TRANSFER_SRC)
    }
}

/// Extension trait providing buffer update methods.
pub trait BufferExt: BufferLike + Sized {
    /// Updates the entire buffer contents.
    ///
    /// Uses direct memory writes for host-visible buffers, or staging transfer otherwise.
    fn update_contents<'a, A: StagingBufferAllocator, E>(
        &'a mut self,
        writer: impl FnOnce(&mut [u8]) -> Result<(), E>,
        encoder: &mut CommandEncoder<'a>,
        staging_allocator: &mut A,
    ) -> Result<(), E>
    where
        E: From<vk::Result>,
    {
        self.update_region(writer, encoder, staging_allocator, 0, self.size())
    }

    /// Async version of [`update_contents`](BufferExt::update_contents).
    fn update_contents_async<'a, A: StagingBufferAllocator, E>(
        &'a mut self,
        writer: impl AsyncFnOnce(&mut [u8]) -> Result<(), E>,
        encoder: &mut CommandEncoder<'a>,
        staging_allocator: &mut A,
    ) -> impl Future<Output = Result<(), E>>
    where
        E: From<vk::Result> + Send + Sync,
    {
        self.update_region_async(writer, encoder, staging_allocator, 0, self.size())
    }

    /// Updates a region of the buffer contents.
    ///
    /// Like [`update_contents`](BufferExt::update_contents), but operates on a specific
    /// byte range within the buffer.
    fn update_region<'a, A: StagingBufferAllocator, E>(
        &'a mut self,
        writer: impl FnOnce(&mut [u8]) -> Result<(), E>,
        encoder: &mut CommandEncoder<'a>,
        staging_allocator: &mut A,
        offset: u64,
        size: u64,
    ) -> Result<(), E>
    where
        E: From<vk::Result>,
    {
        if let Some(slice) = self.as_slice_mut() {
            writer(&mut slice[offset as usize..(offset + size) as usize])?;
            Ok(())
        } else {
            let mut staging_buffer = staging_allocator.allocate_staging_buffer(size)?;
            let staging_slice = staging_buffer
                .as_slice_mut()
                .expect("Staging buffer allocator must return a host-visible buffer!");
            writer(staging_slice)?;

            let staging_buffer = encoder.retain(staging_buffer);
            encoder.copy_buffer_region(staging_buffer, 0, self, offset, size);
            Ok(())
        }
    }

    /// Async version of [`update_region`](BufferExt::update_region).
    #[must_use]
    fn update_region_async<'a, A: StagingBufferAllocator, E>(
        &'a mut self,
        writer: impl AsyncFnOnce(&mut [u8]) -> Result<(), E>,
        encoder: &mut CommandEncoder<'a>,
        staging_allocator: &mut A,
        offset: u64,
        size: u64,
    ) -> impl Future<Output = Result<(), E>>
    where
        E: From<vk::Result> + Send + Sync,
    {
        async move {
            if let Some(slice) = self.as_slice_mut() {
                writer(&mut slice[offset as usize..(offset + size) as usize]).await?;
                Ok(())
            } else {
                let mut staging_buffer = staging_allocator.allocate_staging_buffer(self.size())?;
                let staging_slice = staging_buffer
                    .as_slice_mut()
                    .expect("Staging buffer allocator must return a host-visible buffer!");
                writer(staging_slice).await?;

                let staging_buffer = encoder.retain(staging_buffer);
                encoder.copy_buffer_region(staging_buffer, 0, self, offset, size);
                Ok(())
            }
        }
    }
}
impl<T> BufferExt for T where T: BufferLike {}
