//! Vulkan buffer abstractions with automatic memory management.
//!
//! This module provides safe wrappers around Vulkan buffers with different memory
//! allocation strategies optimized for various GPU architectures and use cases.
//!
//! # Choosing an Allocation Strategy
//!
//! - **[`Buffer::new_device`]**: GPU-exclusive memory. Use for render targets, scratch
//!   buffers, and any data generated entirely on the GPU.
//!
//! - **[`Buffer::new_upload`]**: Device-local memory that may be directly writable.
//!   Avoids staging copies on GPUs with resizable bars or integrated GPUs. Write through
//!   [`as_slice_mut`](BufferLike::as_slice_mut) when it returns `Some`, and copy from a
//!   staging buffer otherwise.
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

use crate::{
    Allocator, Device, HasDevice,
    command::{CommandEncoder, GPURef, GPURefMut, NoHostMapping},
    tracking::Access,
    utils::AsVkHandle,
};

/// Common interface for Vulkan buffer types.
///
/// This trait abstracts over different buffer implementations ([`Buffer`],
/// [`RingBufferSuballocation`]) providing a unified interface
/// for accessing buffer properties and memory-mapped data.
///
/// A value of a `BufferLike` type owns its range of memory the way a `Vec<u8>` owns its
/// bytes: reading the bytes needs `&self`, and writing them needs exclusive access, whether
/// the writer is the host (`&mut self`) or the GPU (a [`GPURefMut`], which
/// [`CommandEncoder::retain`] and [`CommandEncoder::lock`] hand out only for values they
/// exclusively control).
///
/// # Memory Access
///
/// Buffers may or may not be host-visible depending on their allocation strategy.
/// Use [`as_slice`](BufferLike::as_slice) and [`as_slice_mut`](BufferLike::as_slice_mut)
/// to access mapped memory when available.
///
/// Mapped memory is never flushed or invalidated: pumicite assumes host-visible memory is
/// also host-coherent.
///
/// # Safety
///
/// Command recording relies on implementations to name only memory they own:
///
/// - The range [`offset()`](Self::offset)`..offset() + `[`size()`](Self::size) of the
///   buffer returned by `vk_handle()` must be a valid range of that buffer, owned by this
///   value alone. No value that isn't part of this one may expose any of that memory, through
///   `BufferLike` or otherwise, while this one is alive. Otherwise a write token for one
///   value would let the GPU write memory another value is reading.
/// - The buffer and its memory must stay valid until this value is dropped.
/// - [`as_slice`](Self::as_slice) and [`as_slice_mut`](Self::as_slice_mut) must return
///   exactly the mapped bytes of that range, or `None`.
/// - `vk_handle()`, `offset()`, `size()` and `device_address()` must not access the
///   buffer's memory. They are called through tokens while the GPU may be writing it.
pub unsafe trait BufferLike:
    AsVkHandle<Handle = vk::Buffer> + Send + Sync + 'static
{
    /// Returns the offset within the underlying buffer.
    ///
    /// For standalone buffers this is always 0. For suballocations (like
    /// [`RingBufferSuballocation`]), this returns the offset within the parent buffer.
    fn offset(&self) -> vk::DeviceSize;

    /// Returns the device address of the first byte of this buffer's range, for use in
    /// shaders. It already includes [`offset()`](Self::offset).
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
}
impl<T: BufferLike + ?Sized> GPURef<'_, T> {
    pub fn offset(&self) -> vk::DeviceSize {
        unsafe { self.unwrap().offset() }
    }
    pub fn device_address(&self) -> vk::DeviceAddress {
        unsafe { self.unwrap().device_address() }
    }
    pub fn size(&self) -> vk::DeviceSize {
        unsafe { self.unwrap().size() }
    }
}
impl<T: BufferLike + ?Sized> GPURefMut<'_, T> {
    pub fn offset(&self) -> vk::DeviceSize {
        unsafe { self.unwrap().offset() }
    }
    pub fn device_address(&self) -> vk::DeviceAddress {
        unsafe { self.unwrap().device_address() }
    }
    pub fn size(&self) -> vk::DeviceSize {
        unsafe { self.unwrap().size() }
    }
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
// Hands out host-mapped memory through `as_slice`.
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
// Safety: a `Buffer` exclusively owns its `VkBuffer` and allocation. They're created by the
// constructors here, or handed over through `Buffer::from_raw`, whose caller guarantees it.
// `Buffer` isn't `Clone`, and it destroys both on drop. The accessors only read fields and
// the allocation's mapping.
unsafe impl BufferLike for Buffer {
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
            let mapped_data = self
                .allocator
                .get_allocation_info(&self.allocation)
                .mapped_data as *const u8;
            if mapped_data.is_null() {
                None
            } else {
                Some(unsafe { std::slice::from_raw_parts(mapped_data, self.size as usize) })
            }
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
    /// Wraps an existing buffer and its allocation.
    ///
    /// # Safety
    ///
    /// - `buffer` must be a valid buffer created from `allocator`'s device with `usage`, at
    ///   least `size` bytes large, and bound to `allocation`, which must come from `allocator`.
    /// - Ownership of both moves to the returned `Buffer`, which destroys them on drop.
    ///   Nothing else may use, wrap or destroy them afterwards, so that the `Buffer` is the
    ///   sole owner [`BufferLike`] requires.
    pub unsafe fn from_raw(
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
    pub fn new_device(
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
                    usage: vk_mem::MemoryUsage::Unknown,
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
                    usage: vk_mem::MemoryUsage::Unknown,
                    flags: vk_mem::AllocationCreateFlags::MAPPED,
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
    /// If [`as_slice_mut`](BufferLike::as_slice_mut) returns `None`, the buffer isn't
    /// host-visible and has to be written through a staging buffer.
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
                    usage: vk_mem::MemoryUsage::Unknown,
                    flags: vk_mem::AllocationCreateFlags::MAPPED,
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
                    usage: vk_mem::MemoryUsage::Unknown,
                    flags: vk_mem::AllocationCreateFlags::MAPPED,
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
// Owns the host mapping its suballocations point into.
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
// Hands out host-mapped memory through `as_slice`.
impl AsVkHandle for RingBufferSuballocation {
    type Handle = vk::Buffer;
    fn vk_handle(&self) -> Self::Handle {
        self.buffer
    }
}
impl RingBufferSuballocation {
    /// Converts this suballocation into a cloneable, read-only handle.
    ///
    /// Behaves like `Arc<RingBufferSuballocation>`, but reuses the reference count
    /// already held on the parent chunk instead of making a new allocation.
    pub fn shared(self) -> SharedRingBufferSuballocation {
        SharedRingBufferSuballocation(self)
    }
}
// Safety: `RingBuffer::allocate_buffer` hands out non-overlapping ranges, only moving its head
// forward within a chunk. It reuses a chunk only once the ring holds the chunk's sole `Arc`,
// i.e. no suballocation from it is alive. Suballocations aren't `Clone`, and
// `SharedRingBufferSuballocation`, which is, doesn't implement `BufferLike`. `ptr` points at
// this range's mapped bytes, and the `Arc` keeps the chunk alive.
unsafe impl BufferLike for RingBufferSuballocation {
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
}

/// A shared, read-only [`RingBufferSuballocation`]. Behaves like [`Arc<RingBufferSuballocation>`]
///
/// Created by [`RingBufferSuballocation::shared`]. Cloning only bumps the parent
/// chunk's reference count. Since clones alias the same memory, it doesn't implement
/// [`BufferLike`]: its contract requires exclusive ownership. To use it on the GPU, retain a
/// clone and call `deref_inner` on the token, which only yields a read-only [`GPURef`].
pub struct SharedRingBufferSuballocation(RingBufferSuballocation);
impl Clone for SharedRingBufferSuballocation {
    fn clone(&self) -> Self {
        let inner = &self.0;
        SharedRingBufferSuballocation(RingBufferSuballocation {
            buffer: inner.buffer,
            offset: inner.offset,
            size: inner.size,
            ptr: inner.ptr,
            device_address: inner.device_address,
            _chunk: inner._chunk.clone(),
        })
    }
}
impl AsVkHandle for SharedRingBufferSuballocation {
    type Handle = vk::Buffer;
    fn vk_handle(&self) -> Self::Handle {
        self.0.buffer
    }
}
impl<'a> GPURef<'a, SharedRingBufferSuballocation> {
    pub fn deref_inner(self) -> GPURef<'a, RingBufferSuballocation> {
        unsafe {
            let this = self.unwrap();
            GPURef::new_unchecked(&this.0)
        }
    }
}
impl<'a> GPURefMut<'a, SharedRingBufferSuballocation> {
    pub fn deref_inner(self) -> GPURef<'a, RingBufferSuballocation> {
        unsafe {
            let this = self.unwrap();
            GPURef::new_unchecked(&this.0)
        }
    }
}
/// A buffer that abstracts over the differences between integrated and discrete GPUs.
/// On integrated GPUs, and on GPUs with memory that is both device-local and host-cached,
/// a single buffer serves both CPU and GPU access. On other discrete GPUs, this uses a host
/// buffer for CPU access and a device buffer for GPU access, with explicit transfers
/// between them.
///
/// Can be treated as DEVICE_LOCAL, HOST_VISIBLE, HOST_CACHED, but non-coherent memory.
///
/// # Usage
///
/// The host accesses the buffer through `&mut ManagedBuffer`, and the GPU through a
/// [`GPURefMut`], usually by keeping the buffer in a [`GPUMutex`](crate::sync::GPUMutex):
///
/// 1. Write data via [`as_slice_mut`](ManagedBuffer::as_slice_mut)
/// 2. Lock the buffer on a command encoder, transition it to [`Access::COPY_WRITE`] and call
///    [`flush`](GPURefMut::<ManagedBuffer>::flush) to make the writes visible to the GPU
/// 3. GPU reads from the buffer returned by
///    [`device_buffer`](GPURefMut::<ManagedBuffer>::device_buffer)
///
/// To read back data written by the GPU, transition the buffer to
/// `Access::COPY_READ | Access::HOST_READ`, call
/// [`invalidate`](GPURefMut::<ManagedBuffer>::invalidate), wait for the command buffer to
/// complete, then read via [`as_slice`](ManagedBuffer::as_slice).
///
/// The caller tracks the state of [`device_buffer`](GPURefMut::<ManagedBuffer>::device_buffer)
/// with a [`ResourceState`](crate::tracking::ResourceState), as for any other buffer.
/// `flush` and `invalidate` on the same buffer in one command buffer also need
/// `encoder.memory_barrier(Access::COPY_WRITE, Access::COPY_READ | Access::COPY_WRITE)`
/// between them, since both may copy through a staging buffer the caller can't track.
pub struct ManagedBuffer {
    /// The buffer the GPU accesses. When `staging` is `None`, it's also the host-visible,
    /// mapped buffer the host accesses.
    device: Buffer,
    /// Host-visible copy of `device`, on GPUs where `device` isn't host-visible.
    staging: Option<Buffer>,
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
            .memory_type_map()
            .dynamic_device_local
        {
            let buffer = Buffer::new_dynamic(allocator, size, alignment, usage)?;
            Ok(Self {
                device: buffer,
                staging: None,
            })
        } else {
            let transfer = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            let staging = Buffer::new_dynamic(allocator.clone(), size, alignment, transfer)?;
            let device = Buffer::new_device(allocator, size, alignment, usage | transfer)?;
            Ok(Self {
                device,
                staging: Some(staging),
            })
        }
    }
    /// Returns `true` if the host and the GPU access a single buffer, so
    /// [`flush`](GPURefMut::<ManagedBuffer>::flush) and
    /// [`invalidate`](GPURefMut::<ManagedBuffer>::invalidate) record no copies.
    pub fn is_direct(&self) -> bool {
        self.staging.is_none()
    }
    fn host(&self) -> &Buffer {
        self.staging.as_ref().unwrap_or(&self.device)
    }
    fn host_mut(&mut self) -> &mut Buffer {
        self.staging.as_mut().unwrap_or(&mut self.device)
    }
    /// Returns the host's view of the buffer.
    ///
    /// GPU writes show up here only after [`invalidate`](GPURefMut::<ManagedBuffer>::invalidate)
    /// and the completion of the command buffer that recorded it.
    pub fn as_slice(&self) -> &[u8] {
        self.host().as_slice().unwrap()
    }
    /// Returns the host's view of the buffer.
    ///
    /// Writes reach the GPU only after [`flush`](GPURefMut::<ManagedBuffer>::flush).
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        self.host_mut().as_slice_mut().unwrap()
    }
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.host_mut().as_mut_ptr()
    }
    pub fn as_ptr(&self) -> *const u8 {
        self.host().as_ptr()
    }
    pub fn allocator(&self) -> &Allocator {
        self.device.allocator()
    }
    pub fn size(&self) -> u64 {
        self.device.size()
    }
}
impl AsVkHandle for ManagedBuffer {
    type Handle = vk::Buffer;

    fn vk_handle(&self) -> Self::Handle {
        self.device.vk_handle()
    }
}
// `ManagedBuffer` doesn't implement `BufferLike`: its `as_slice` returns the staging copy,
// not the mapped bytes of the buffer the GPU accesses. Commands use the buffer projected
// by `device_buffer` instead.
impl<'a> GPURef<'a, ManagedBuffer> {
    /// Returns the buffer the GPU reads.
    pub fn device_buffer(self) -> GPURef<'a, Buffer> {
        // Safety: `device` is part of the `ManagedBuffer`, so this token guarantees for it
        // what it guarantees for the whole value.
        unsafe { GPURef::new_unchecked(&self.unwrap().device) }
    }
}
impl<'a> GPURefMut<'a, ManagedBuffer> {
    /// Returns the buffer the GPU reads and writes.
    pub fn device_buffer(self) -> GPURefMut<'a, Buffer> {
        // Safety: `device` is part of the `ManagedBuffer`, so this token guarantees for it
        // what it guarantees for the whole value.
        unsafe { GPURefMut::new_unchecked(&self.unwrap().device) }
    }

    /// Makes host writes through [`as_slice_mut`](ManagedBuffer::as_slice_mut) visible to
    /// the GPU commands recorded after this call.
    ///
    /// On discrete GPUs, this copies the staging buffer into
    /// [`device_buffer`](Self::device_buffer). On integrated GPUs, it records nothing: the
    /// host writes the buffer the GPU reads, and submitting the command buffer makes those
    /// writes visible.
    ///
    /// # Synchronization
    ///
    /// The caller synchronizes [`device_buffer`](Self::device_buffer) as for any copy into
    /// it: call [`use_resource`](CommandEncoder::use_resource) with [`Access::COPY_WRITE`]
    /// before this, and with the access of the commands that read the buffer after it. The
    /// barriers may stay pending: this emits them before the copy.
    ///
    /// The caller also orders this against [`invalidate`](Self::invalidate) on the same
    /// buffer in the same command buffer, whose copy writes the staging buffer this copy
    /// reads. Record this between them:
    ///
    /// ```ignore
    /// encoder.memory_barrier(Access::COPY_WRITE, Access::COPY_READ | Access::COPY_WRITE);
    /// ```
    ///
    /// Command buffers that lock the buffer are already ordered by its
    /// [`GPUMutex`](crate::sync::GPUMutex).
    pub fn flush(self, encoder: &mut CommandEncoder<'a>) {
        // Safety: only used to reach the staging buffer, whose memory isn't accessed.
        let Some(staging) = (unsafe { &self.unwrap().staging }) else {
            return;
        };
        // Safety: `staging` is part of the `ManagedBuffer`, so this token guarantees for it
        // what it guarantees for the whole value.
        let staging: GPURef<'a, Buffer> = unsafe { GPURef::new_unchecked(staging) };
        encoder.emit_barriers();
        encoder.copy_buffer(staging, self.device_buffer());
    }

    /// Makes GPU writes to [`device_buffer`](Self::device_buffer) recorded before this call
    /// visible to [`as_slice`](ManagedBuffer::as_slice), once this command buffer completes.
    ///
    /// On discrete GPUs, this copies [`device_buffer`](Self::device_buffer) into the staging
    /// buffer. On integrated GPUs, it records nothing: the host reads the buffer the GPU
    /// writes.
    ///
    /// # Synchronization
    ///
    /// The caller synchronizes [`device_buffer`](Self::device_buffer): call
    /// [`use_resource`](CommandEncoder::use_resource) with
    /// `Access::COPY_READ | Access::HOST_READ` before this, so the writes are visible to the
    /// copy on discrete GPUs and to the host on integrated GPUs. The barriers may stay
    /// pending: this emits them before the copy, and recording emits them at the end. This
    /// makes the copy's writes to the staging buffer visible to the host itself.
    ///
    /// The caller also orders this against [`flush`](Self::flush) and other `invalidate`s on
    /// the same buffer in the same command buffer, whose copies access the staging buffer
    /// this copy writes. Record this between them:
    ///
    /// ```ignore
    /// encoder.memory_barrier(Access::COPY_WRITE, Access::COPY_READ | Access::COPY_WRITE);
    /// ```
    ///
    /// Command buffers that lock the buffer are already ordered by its
    /// [`GPUMutex`](crate::sync::GPUMutex).
    pub fn invalidate(self, encoder: &mut CommandEncoder<'a>) {
        // Safety: only used to reach the staging buffer, whose memory isn't accessed.
        let Some(staging) = (unsafe { &self.unwrap().staging }) else {
            return;
        };
        // Safety: `staging` is part of the `ManagedBuffer`, so this token guarantees for it
        // what it guarantees for the whole value.
        let staging: GPURefMut<'a, Buffer> = unsafe { GPURefMut::new_unchecked(staging) };
        encoder.emit_barriers();
        encoder.copy_buffer(self.device_buffer(), staging);
        // Semaphore waits on the host don't make GPU writes visible to it.
        encoder.memory_barrier(Access::COPY_WRITE, Access::HOST_READ);
    }
}

/// Trait for types that can allocate staging buffers.
///
/// Staging buffers are temporary host-visible buffers used to transfer data
/// to device-local memory. This trait is implemented by [`RingBuffer`] (for
/// efficient transient allocations) and [`Allocator`] (for standalone buffers).
///
/// Used by `AsyncTransfer::update_image` in `bevy_pumicite`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{command::CommandPool, sync::GPUMutex, sync::Timeline, tracking::ResourceState};

    /// Writes on the host, flushes, overwrites a prefix on the GPU, then invalidates.
    fn managed_buffer_round_trip(direct: bool) {
        let (device, mut queue) = Device::create_system_default().unwrap();
        let allocator = Allocator::new(device.clone()).unwrap();
        let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
        let mut timeline = Timeline::new(device).unwrap();

        let size = 64;
        let usage = vk::BufferUsageFlags::TRANSFER_DST;
        let mut buffer = if direct {
            ManagedBuffer {
                device: Buffer::new_dynamic(allocator, size, 4, usage).unwrap(),
                staging: None,
            }
        } else {
            let transfer = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
            ManagedBuffer {
                staging: Some(Buffer::new_dynamic(allocator.clone(), size, 4, transfer).unwrap()),
                device: Buffer::new_device(allocator, size, 4, usage | transfer).unwrap(),
            }
        };
        assert_eq!(buffer.is_direct(), direct);
        for (i, byte) in buffer.as_slice_mut().iter_mut().enumerate() {
            *byte = i as u8;
        }
        let buffer = GPUMutex::new(buffer);

        let mut cmd = pool.alloc().unwrap();
        timeline.schedule(&mut cmd);
        pool.begin(&mut cmd).unwrap();
        pool.record(&mut cmd, |encoder| {
            let mut state = ResourceState::default();
            let locked = encoder.lock(&buffer, vk::PipelineStageFlags2::ALL_COMMANDS);
            encoder.use_resource::<Buffer>(&mut state, Access::COPY_WRITE);
            locked.flush(encoder);
            encoder.use_resource::<Buffer>(&mut state, Access::COPY_WRITE);
            encoder.emit_barriers();
            encoder.update_buffer(locked.device_buffer(), &[0xff; 8]);
            encoder.use_resource::<Buffer>(&mut state, Access::COPY_READ | Access::HOST_READ);
            encoder.memory_barrier(Access::COPY_WRITE, Access::COPY_READ | Access::COPY_WRITE);
            locked.invalidate(encoder);
        });
        pool.finish(&mut cmd).unwrap();
        queue.submit(&mut cmd).unwrap();
        cmd.block_until_completion().unwrap();

        let buffer = buffer.unwrap_block().unwrap();
        let data = buffer.as_slice();
        assert!(data[..8].iter().all(|&b| b == 0xff));
        assert!(
            data[8..]
                .iter()
                .enumerate()
                .all(|(i, &b)| b == (i + 8) as u8)
        );
        pool.free(cmd);
    }

    #[test]
    fn managed_buffer_round_trip_staging() {
        managed_buffer_round_trip(false);
    }

    #[test]
    fn managed_buffer_round_trip_direct() {
        managed_buffer_round_trip(true);
    }
}
