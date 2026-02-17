//! Automatic resource state tracking and pipeline barrier generation.
//!
//! This module provides the foundation for Pumicite's automatic synchronization system.
//! Instead of manually inserting `vkCmdPipelineBarrier` calls, you track resource states
//! and let the framework compute minimal barriers for you.
//!
//! # Overview
//!
//! Vulkan requires explicit synchronization between GPU operations. You must call vkCmdPipelineBarrier2:
//! - Insert **execution barriers** to order operations between [pipeline stages](`vk::PipelineStageFlags`)
//! - Specify **memory barriers** to make writes [accesses](vk::AccessFlags2) visible to later reads [accesses](vk::AccessFlags2) (memory dependency)
//! - Transition **image layouts** for different usage patterns
//! - Transfer ownership between different queue families
//!
//! This module offers opt-in automation for these requirements through:
//! - [`ResourceState`] - Tracks a resource's access history
//! - [`Access`] - Describes how a resource is used (stage + access mask)
//! - [`CommandEncoder::use_resource`](crate::command::CommandEncoder::use_resource) - Automatically computes and accumulates barriers
//!
//! # Quick Start
//!
//! ```
//! # use pumicite::{Device, command::CommandPool, sync::Timeline};
//! use pumicite::tracking::{ResourceState, Access};
//! # let (device, queue) = Device::create_system_default().unwrap();
//! # let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
//! # let mut timeline = Timeline::new(device).unwrap();
//! # let mut cmd = pool.alloc().unwrap();
//! # timeline.schedule(&mut cmd);
//! # pool.begin(&mut cmd).unwrap();
//!
//! // Track a buffer's state
//! let mut buffer_state = ResourceState::default();
//!
//! pool.record(&mut cmd, |encoder| {
//!     // Declare write access - no barrier needed (first use)
//!     encoder.use_resource::<()>(&mut buffer_state, Access::COMPUTE_WRITE);
//!     encoder.emit_barriers();
//!     // ... dispatch compute work ...
//!
//!     // Declare read access - barrier automatically computed
//!     encoder.use_resource::<()>(&mut buffer_state, Access::VERTEX_READ);
//!     encoder.emit_barriers(); // Inserts COMPUTE_WRITE → VERTEX_READ barrier
//!     // ... draw calls using the buffer ...
//! });
//! # pool.finish(&mut cmd).unwrap();
//! ```
//!
//! # Key Concepts
//!
//! ## Access Patterns
//!
//! An [`Access`] describes both **when** [`vk::PipelineStageFlags`] and **how** [`vk::AccessFlags2`] a resource is used:
//!
//! ```rust,no_run
//! # use pumicite::tracking::Access;
//! # use pumicite::ash::vk;
//! // Reading from a vertex buffer
//! let vertex_read = Access::VERTEX_READ;
//!
//! // Writing to an image from a compute shader
//! let compute_write = Access {
//!     stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
//!     access: vk::AccessFlags2::SHADER_WRITE,
//! };
//! ```
//!
//! ## Resource States
//!
//! A [`ResourceState`] remembers:
//! - The last **write** operation
//! - All stages **reading** from the resource since that write
//! - Current **image layout**
//! - Current owning queue family
//!
//! Store it with your resources:
//!
//! ```no_run
//! use pumicite::image::Image;
//! use pumicite::sync::GPUMutex;
//! use pumicite::tracking::ResourceState;
//!
//! struct RenderTarget {
//!     image: GPUMutex<Image>,
//!     state: ResourceState,  // Persist across systems and frames
//! }
//! ```
//!
//! ## Barrier Batching
//!
//! Barriers are accumulated and emitted in batches for efficiency:
//!
//! ```
//! # use pumicite::{Device, command::CommandPool, sync::Timeline};
//! use pumicite::tracking::{ResourceState, Access};
//! # let (device, queue) = Device::create_system_default().unwrap();
//! # let mut pool = CommandPool::new(device.clone(), queue.family_index()).unwrap();
//! # let mut timeline = Timeline::new(device).unwrap();
//! # let mut cmd = pool.alloc().unwrap();
//! # timeline.schedule(&mut cmd);
//! # pool.begin(&mut cmd).unwrap();
//! let mut state1 = ResourceState::default();
//! let mut state2 = ResourceState::default();
//! pool.record(&mut cmd, |encoder| {
//!     // Accumulate multiple barriers
//!     encoder.use_resource::<()>(&mut state1, Access::COMPUTE_WRITE);
//!     encoder.use_resource::<()>(&mut state2, Access::COMPUTE_WRITE);
//!
//!     // Single vkCmdPipelineBarrier2 call for all accumulated barriers
//!     encoder.emit_barriers();
//! });
//! # pool.finish(&mut cmd).unwrap();
//! ```
//!
//! ## Image Layout Transitions
//!
//! Use [`CommandEncoder::use_image_resource`](crate::command::CommandEncoder::use_image_resource)
//! for images to automatically handle layout transitions:
//!
//! ```no_run
//! use pumicite::prelude::*;
//! use pumicite::tracking::{ResourceState, Access};
//!
//! fn example<'a>(encoder: &mut CommandEncoder<'a>, image: &'a Image, state: &mut ResourceState) {
//!     // Transition from TRANSFER_DST_OPTIMAL to SHADER_READ_ONLY_OPTIMAL
//!     encoder.use_image_resource(
//!         image,
//!         state,
//!         Access::FRAGMENT_SAMPLED_READ,
//!         vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
//!         0..1,  // mip levels
//!         0..1,  // array layers
//!         false, // don't discard previous content
//!     );
//! }
//! ```

use std::{
    fmt::Debug,
    ops::{BitOr, BitOrAssign},
};

use ash::vk;

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct Access {
    pub stage: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
}
impl BitOr for Access {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            stage: self.stage | rhs.stage,
            access: self.access | rhs.access,
        }
    }
}
impl BitOrAssign for Access {
    fn bitor_assign(&mut self, rhs: Self) {
        self.stage |= rhs.stage;
        self.access |= rhs.access;
    }
}

impl Access {
    pub const NONE: Access = Access {
        stage: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
    };
    pub const VERTEX_READ: Access = Access {
        stage: vk::PipelineStageFlags2::VERTEX_INPUT,
        access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
    };
    pub const VERTEX_SAMPLED_READ: Access = Access {
        stage: vk::PipelineStageFlags2::VERTEX_SHADER,
        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
    };
    pub const COPY_READ: Access = Access {
        stage: vk::PipelineStageFlags2::COPY,
        access: vk::AccessFlags2::TRANSFER_READ,
    };
    pub const FRAGMENT_SAMPLED_READ: Access = Access {
        stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
    };
    pub const COPY_WRITE: Access = Access {
        stage: vk::PipelineStageFlags2::COPY,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    };
    pub const BLIT_DST: Access = Access {
        stage: vk::PipelineStageFlags2::BLIT,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    };
    pub const BLIT_SRC: Access = Access {
        stage: vk::PipelineStageFlags2::BLIT,
        access: vk::AccessFlags2::TRANSFER_READ,
    };
    pub const CLEAR: Access = Access {
        stage: vk::PipelineStageFlags2::CLEAR,
        access: vk::AccessFlags2::TRANSFER_WRITE,
    };
    pub const COLOR_ATTACHMENT_WRITE: Access = Access {
        stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    };
    pub const EARLY_FRAGMENT_TEST_WRITE: Access = Access {
        stage: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
    };
    pub const RTX_WRITE: Access = Access {
        stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        access: vk::AccessFlags2::SHADER_WRITE,
    };
    pub const RTX_READ: Access = Access {
        stage: vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
        access: vk::AccessFlags2::SHADER_READ,
    };
    pub const COMPUTE_WRITE: Access = Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_WRITE,
    };
    pub const COMPUTE_READ: Access = Access {
        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        access: vk::AccessFlags2::SHADER_READ,
    };
    pub const ACCELERATION_STRUCTURE_BUILD_READ: Access = Access {
        stage: vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
        access: vk::AccessFlags2::SHADER_READ,
    };
    pub const ALL_COMMANDS: Access = Access {
        stage: vk::PipelineStageFlags2::ALL_COMMANDS,
        access: vk::AccessFlags2::from_raw(
            vk::AccessFlags2::MEMORY_READ.as_raw() | vk::AccessFlags2::MEMORY_WRITE.as_raw(),
        ),
    };
    pub fn is_writeonly(&self) -> bool {
        if self.access == vk::AccessFlags2::empty() {
            return false;
        }
        // Clear all the write bits. If nothing is left, that means there's no read bits.
        self.access & !utils::ALL_WRITE_BITS == vk::AccessFlags2::NONE
    }

    pub fn is_readonly(&self) -> bool {
        if self.access == vk::AccessFlags2::empty() {
            return false;
        }
        // Clear all the read bits. If nothing is left, that means there's no write bits.
        self.access & !utils::ALL_READ_BITS == vk::AccessFlags2::NONE
    }
}

/// Tracks the current state of a Vulkan resource across systems and frames.
///
/// `ResourceState` enables automatic pipeline barrier insertion by tracking how a resource was
/// last accessed (pipeline stage, access flags, image layout, and queue family). When you request
/// a new access pattern via [use_resource](`crate::command::CommandEncoder::use_resource`),
/// [use_image_resource](`crate::command::CommandEncoder::use_image_resource`),
/// or [use_buffer_resource](`crate::command::CommandEncoder::use_buffer_resource`), the state automatically
/// computes the minimal synchronization required.
///
/// # Why ResourceState is Needed
///
/// When creating a rendering system, you have full knowledge of what's happening inside that system,
/// so you can correctly specify barriers between your renders or dispatches using
/// `vkCmdPipelineBarrier`. However, this local reasoning falls apart when you need to reason
/// about resource states globally, because you have no knowledge of what could happen before or
/// after your render system. Another developer can insert a system between your render system
/// and the system that you assumed will run after, for example.
///
/// `ResourceState` gives you the ability to reason about resource states globally by maintaining
/// the last known state of the resource, allowing each system to transition from that known state
/// to the state it needs.
///
/// # Core Concept
///
/// In Vulkan, you must manually insert pipeline barriers to:
/// 1. **Synchronize execution** - ensure ["before" pipeline stages](`vk::PipelineStageFlags2`) in operation A
///    finishes before ["after" pipeline stages](`vk::PipelineStageFlags2`) in operation B starts
/// 2. **Make memory visible** - flush caches so writes with ["before" access](`vk::AccessFlags2`) from A are
///    visible to reads with ["after" access](`vk::AccessFlags2`) in B
/// 3. **Transition image layouts** - convert images between different [`vk::ImageLayout`]
/// 4. **Transfer queue family ownership** - for resources created with [`vk::SharingMode::EXCLUSIVE`].
///
/// `ResourceState` automates this by remembering:
/// - The **last write** to the resource (when and how it was written)
/// - All **pending reads** since that write (what stages have read it)
/// - The **current image layout** (for images only)
/// - The **owning queue family** (for multi-queue rendering)
///
/// When you declare a new access, it compares the old state with the new state and generates
/// the appropriate barrier - or no barrier if synchronization isn't needed.
///
/// # Storage Patterns
///
/// Graphics APIs that offer automatic syncronization typically stores tracking info centrally, with mapping to resource IDs or
/// alongside the resource itself. This creates challenges with multi-thread command buffer recording and is also wasteful for
/// many applications where the vast majority of assets are readonly. With `ResourceState` you're responsible for deciding whether
/// you want tracking and decide where to store the resource state.
///
/// ## Pattern 1: Long-lived Resources
///
/// Store `ResourceState` alongside resources that persist across multiple frames or systems:
///
/// ```no_run
/// use pumicite::image::Image;
/// use pumicite::sync::GPUMutex;
/// use pumicite::tracking::ResourceState;
///
/// // In a resource struct or ECS component
/// struct GBufferTexture {
///     depth: GPUMutex<Image>,
///     state: ResourceState,  // Tracks state across all systems and frames
/// }
/// ```
///
/// ## Pattern 2: Local / Stack storage for Transient Resources (Use with Caution)
///
/// Store state on the stack if the resource is only used locally, and you only need to
/// reason about resource states locally.
///
/// ```no_run
/// use pumicite::prelude::*;
/// use pumicite::tracking::ResourceState;
///
/// fn local_system<'a>(encoder: &mut CommandEncoder<'a>, temp_image: Image) {
///     let temp_image = encoder.retain(temp_image);
///     let mut state = ResourceState::default();
///
///     // Use only within this function
///     encoder.use_image_resource(temp_image, &mut state,
///         pumicite::tracking::Access::CLEAR, vk::ImageLayout::GENERAL, 0..1, 0..1, false);
///     encoder.emit_barriers();
///     // encoder.dispatch(/* ... */);
///
///     encoder.use_image_resource(temp_image, &mut state,
///         pumicite::tracking::Access::CLEAR, vk::ImageLayout::GENERAL, 0..1, 0..1, false);
///     encoder.emit_barriers();
///     // encoder.dispatch(/* ... */);
/// }
/// // ⚠️ State is lost here! If another system uses temp_image later,
/// // it won't know about the previous access.
/// ```
///
/// **Use this when:**
/// - Resource is transient and will be dropped after the scope ends
/// - Resource is exclusively written in this scope and becomes read-only after the scope ends
/// - There is a reasonable, well-documented assumption about resource state when used elsewhere.
///   For example, this texture will always be accessed in [`vk::ImageLayout::READ_ONLY_OPTIMAL`] layout
///   by the graphics queue family, and nobody will ever write into it after this function returns.
///
///
/// # Tracking Granularity: Whole Resources vs. Subresources
///
/// ## Whole Resource Tracking (Start here)
///
/// One `ResourceState` per entire resource:
///
/// ```no_run
/// use pumicite::image::Image;
/// use pumicite::sync::GPUMutex;
/// use pumicite::tracking::ResourceState;
///
/// struct Texture {
///     image: GPUMutex<Image>,  // 2048x2048, 11 mip levels
///     state: ResourceState,  // Tracks ALL mip levels together
/// }
/// ```
///
/// - Simple and low overhead
/// - Coarse-grained: accessing mip 0 affects tracking for all mips
/// - Oversyncronization if different mips are used independently
///
/// It is also possible to track multiple resources with one state if they will always be used together:
/// ```no_run
/// use pumicite::image::Image;
/// use pumicite::sync::GPUMutex;
/// use pumicite::tracking::ResourceState;
///
/// struct GBufferTextures {
///     depth: Image,
///     normal: Image,
///     albedo: Image,
///     specular: Image,
/// }
/// struct GBuffer {
///     textures: GPUMutex<GBufferTextures>,
///     state: ResourceState,
/// }
/// ```
///
/// ## Fine-Grained Subresource Tracking
///
/// Track individual subresources (mip levels, array layers, buffer regions) separately:
///
/// ```no_run
/// use pumicite::image::Image;
/// use pumicite::tracking::ResourceState;
///
/// struct MipChainTexture {
///     image: Image,
///     per_mip_states: Vec<ResourceState>,  // One state per mip level
/// }
/// ```
///
/// **Pros:**
/// - Precise synchronization
/// - Useful for cases where each level (or subresource) is accessed independently
/// - Can reduce false dependencies
///
/// **Cons:**
/// - Higher CPU overhead (more state to track and update)
/// - More complex to manage
///
/// **Possible scenarios to use fine-grained tracking:**
/// - Mipmap generation pipelines
/// - Sparse or tiled resources
/// - Large buffers where different regions are accessed by different stages
/// - Profile first! Fine-grained tracking may not be worth the CPU overhead.
///
/// # Submission Boundaries and Lifetimes
///
/// `ResourceState` is designed for tracking within **a single submission** or across **multiple frames**.
/// However, there are important semantics to understand:
///
/// ## Within a Single Submission
///
/// The state accumulates reads and optimizes barriers:
///
/// ```no_run
/// use pumicite::prelude::*;
/// use pumicite::tracking::{ResourceState, Access};
///
/// fn example<'a>(encoder: &mut CommandEncoder<'a>, buffer: &'a impl pumicite::buffer::BufferLike) {
///     let mut state = ResourceState::default();
///
///     // Write to buffer in compute shader
///     encoder.use_resource::<()>(&mut state, Access::COMPUTE_WRITE);
///     encoder.emit_barriers();
///
///     // Read in vertex shader - needs barrier from COMPUTE_WRITE
///     encoder.use_resource::<()>(&mut state, Access::VERTEX_READ);
///     encoder.emit_barriers();
///
///     // Read-after-read in fragment shader - NO barrier needed!
///     // Vertex shader is earlier than fragment shader, so data is already available
///     encoder.use_resource::<()>(&mut state, Access::FRAGMENT_SAMPLED_READ);
///     // No barrier emitted
/// }
/// ```
///
/// ## Across Submissions
///
/// When command buffers are submitted separately, ensure proper semaphore synchronization.
/// `ResourceState` tracks access patterns, but **cross-submission synchronization requires
/// semaphores** (handled by Timeline and GPUMutex):
///
/// ```no_run
/// use pumicite::prelude::*;
/// use pumicite::tracking::{ResourceState, Access};
///
/// // Frame 1: Write to texture
/// fn frame1(encoder: &mut CommandEncoder, texture: &GPUMutex<Image>, state: &mut ResourceState) {
///     let image = encoder.lock(texture, vk::PipelineStageFlags2::FRAGMENT_SHADER);
///     encoder.use_image_resource(image, state,
///         Access::COLOR_ATTACHMENT_WRITE, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, 0..1, 0..1, false);
/// }
/// // Submit happens here, semaphore signals
///
/// // Frame 2: Read from texture
/// fn frame2(encoder: &mut CommandEncoder, texture: &GPUMutex<Image>, state: &mut ResourceState) {
///     // encoder.lock() automatically inserts semaphore wait (command.rs:303-337)
///     let image = encoder.lock(texture, vk::PipelineStageFlags2::FRAGMENT_SHADER);
///     encoder.use_image_resource(image, state,
///         Access::FRAGMENT_SAMPLED_READ, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, 0..1, 0..1, false);
/// }
/// ```
///
/// **Key point:** `ResourceState` works with `GPUMutex` and Timeline semaphores to handle
/// cross-submission synchronization. The state tracks access patterns; semaphores enforce ordering.
///
/// ## Resetting State
///
/// You generally **don't** need to reset `ResourceState`. It accumulates correctly across
/// submissions. However, you may want to reset for:
///
/// - **Discarding content:** Set to default and use `discard_content: true` to avoid
///   unnecessary cache flushes
/// - **Debugging:** Reset to known state to isolate issues
///
/// ```rust,no_run
/// # use pumicite::tracking::ResourceState;
/// # let mut state = ResourceState::default();
/// // Clear state when you don't care about previous content
/// state = ResourceState::default();
/// ```
#[derive(Clone, PartialEq, Eq, Default)]
pub struct ResourceState {
    /// The pipeline stage and access flags of the most recent write operation.
    ///
    /// When a new access is requested, this is used as the source of the pipeline barrier.
    /// For read operations, this ensures writes are visible. For write operations, this
    /// prevents WAW (Write-After-Write) hazards.
    pub write: Access,

    /// Accumulated pipeline stages that have read from the resource since the last write.
    ///
    /// This enables an optimization: if a new read happens at a stage later than all previous
    /// reads, no barrier is needed (data is already visible). Tracks the "earliest" read stage
    /// to minimize redundant execution barriers.
    ///
    /// Reset to empty when a new write occurs.
    pub reads: vk::PipelineStageFlags2,

    /// Current image layout (for images only).
    ///
    /// Image layout transitions require image-specific barriers. When the requested layout
    /// differs from this value, `use_image_resource()` creates a layout transition barrier.
    /// For buffers, this field is unused.
    pub layout: vk::ImageLayout,

    /// Queue family that currently owns the resource.
    pub queue_family: u32,
}
impl Debug for ResourceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("ResourceState");
        debug.field("write_stage", &self.write.stage);
        debug.field("write_access", &self.write.access);
        debug.field("pending_read_stages", &self.reads);
        if self.layout != vk::ImageLayout::default() {
            debug.field("image_layout", &self.layout);
        }
        debug.field("queue_family", &self.queue_family);
        debug.finish()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct MemoryBarrier {
    pub src: Access,
    pub dst: Access,
}
impl BitOr for MemoryBarrier {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            src: self.src | rhs.src,
            dst: self.dst | rhs.dst,
        }
    }
}
impl BitOrAssign for MemoryBarrier {
    fn bitor_assign(&mut self, rhs: Self) {
        self.src |= rhs.src;
        self.dst |= rhs.dst;
    }
}

impl ResourceState {
    /// Create a new state with a presumed initial state
    pub fn new(access: Access) -> Self {
        Self {
            write: access,
            reads: vk::PipelineStageFlags2::empty(),
            layout: vk::ImageLayout::UNDEFINED,
            queue_family: 0,
        }
    }
    /// Create a new state with a presumed initial state and image layout
    pub fn new_with_image_layout(access: Access, layout: vk::ImageLayout) -> Self {
        Self {
            write: access,
            reads: vk::PipelineStageFlags2::empty(),
            layout,
            queue_family: 0,
        }
    }
    /// Computes the minimal pipeline barrier needed to transition from the current state to a new access pattern.
    ///
    /// This is the core algorithm that enables automatic barrier insertion. It analyzes the
    /// current state and the requested access to determine what synchronization is required.
    ///
    /// # Parameters
    ///
    /// - `next`: The requested access pattern (stage + access flags)
    /// - `with_layout_transition`: Whether an image layout change is occurring
    ///
    /// # Returns
    ///
    /// A `MemoryBarrier` containing the source and destination pipeline stages and access flags.
    /// If no synchronization is needed, returns an empty barrier (all fields are NONE/empty).
    ///
    /// # Algorithm Overview
    ///
    /// 1. **First use** (line 472-479): Resource never accessed before → no barrier needed
    ///    (Vulkan allows undefined initial state)
    ///
    /// 2. **Read-after-read optimization** (line 480-492): If the new access is a read and
    ///    happens at a later pipeline stage than previous reads, the data is already visible
    ///    from the write→first-read barrier. No additional barrier needed.
    ///
    /// 3. **Write-after-read** (line 494-507): Inserting a write after reads requires waiting
    ///    for all reads to complete (WAR hazard). This only needs an execution dependency,
    ///    not a memory barrier, unless a layout transition is involved.
    ///
    /// 4. **Write-after-write** (default case): Full memory barrier to ensure previous write
    ///    is flushed and visible to the new write (WAW hazard).
    ///
    /// After computing the barrier, the state is updated:
    /// - For reads: accumulate into `self.reads` (tracking earliest read stage)
    /// - For writes: record as `self.write` and clear `self.reads`
    pub(crate) fn transition(
        &mut self,
        next: Access,
        with_layout_transition: bool,
    ) -> MemoryBarrier {
        let mut barrier = MemoryBarrier {
            src: self.write,
            dst: next,
        };
        if self.write == Default::default() {
            // Resource was never accessed before.
            if with_layout_transition {
                barrier.src = Access::default();
            } else {
                barrier = MemoryBarrier::default();
            }
        } else if next.is_readonly() && !with_layout_transition {
            if let Some(ordering) = utils::compare_pipeline_stages(self.reads, next.stage) {
                if ordering.is_gt() {
                    barrier.src.stage = self.reads;
                    barrier.src.access = vk::AccessFlags2::empty();
                    barrier.dst.access = vk::AccessFlags2::empty();
                } else {
                    // it has already been made visible at the desired stage
                    barrier = MemoryBarrier::default();
                }
            }
        } else {
            // The next stage will be writing.
            if self.reads != vk::PipelineStageFlags2::empty() {
                // This is a WAR hazard, which you would usually only need an execution dependency for.
                // meaning you wouldn't need to supply any memory barriers.
                barrier.src.stage = self.reads;
                barrier.src.access = vk::AccessFlags2::empty();
                if !with_layout_transition {
                    // When we do have a layout transition, you still need a memory barrier,
                    // but you don't need any access types in the src access mask. The layout transition
                    // itself is considered a write operation though, so you do need the destination
                    // access mask to be correct - or there would be a WAW hazard between the layout
                    // transition and the color attachment write.
                    barrier.dst.access = vk::AccessFlags2::empty();
                }
            }
        }
        // the info is now available for all stages after next.stage, but not for stages before next.stage.
        if next.is_readonly() {
            self.reads = utils::earlier_stage(self.reads, next.stage);
        } else {
            self.write = next;
            self.reads = vk::PipelineStageFlags2::empty();
        }
        barrier
    }
}

mod utils {
    use ash::vk;
    use std::cmp::Ordering;
    pub const ALL_WRITE_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_WRITE.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::TRANSFER_WRITE.as_raw()
            | vk::AccessFlags2::HOST_WRITE.as_raw()
            | vk::AccessFlags2::MEMORY_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw()
            | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::VIDEO_ENCODE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT.as_raw()
            | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR.as_raw()
            | vk::AccessFlags2::MICROMAP_WRITE_EXT.as_raw()
            | vk::AccessFlags2::OPTICAL_FLOW_WRITE_NV.as_raw(),
    );
    pub const ALL_READ_BITS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
        vk::AccessFlags2::INDIRECT_COMMAND_READ.as_raw()
            | vk::AccessFlags2::INDEX_READ.as_raw()
            | vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw()
            | vk::AccessFlags2::UNIFORM_READ.as_raw()
            | vk::AccessFlags2::INPUT_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::SHADER_READ.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::TRANSFER_READ.as_raw()
            | vk::AccessFlags2::HOST_READ.as_raw()
            | vk::AccessFlags2::MEMORY_READ.as_raw()
            | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
            | vk::AccessFlags2::VIDEO_DECODE_READ_KHR.as_raw()
            | vk::AccessFlags2::VIDEO_ENCODE_READ_KHR.as_raw()
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_READ_EXT.as_raw()
            | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT.as_raw()
            | vk::AccessFlags2::COMMAND_PREPROCESS_READ_NV.as_raw()
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR.as_raw()
            | vk::AccessFlags2::FRAGMENT_DENSITY_MAP_READ_EXT.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT.as_raw()
            | vk::AccessFlags2::DESCRIPTOR_BUFFER_READ_EXT.as_raw()
            | vk::AccessFlags2::INVOCATION_MASK_READ_HUAWEI.as_raw()
            | vk::AccessFlags2::SHADER_BINDING_TABLE_READ_KHR.as_raw()
            | vk::AccessFlags2::MICROMAP_READ_EXT.as_raw()
            | vk::AccessFlags2::OPTICAL_FLOW_READ_NV.as_raw(),
    );
    const GRAPHICS_PIPELINE_ORDER: [vk::PipelineStageFlags2; 13] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::INDEX_INPUT,
        vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT,
        vk::PipelineStageFlags2::VERTEX_SHADER,
        vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
        vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
        vk::PipelineStageFlags2::GEOMETRY_SHADER,
        vk::PipelineStageFlags2::TRANSFORM_FEEDBACK_EXT,
        vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ];
    const GRAPHICS_MESH_PIPELINE_ORDER: [vk::PipelineStageFlags2; 8] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::TASK_SHADER_EXT,
        vk::PipelineStageFlags2::MESH_SHADER_EXT,
        vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
    ];
    const COMPUTE_PIPELINE_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
    ];
    const FRAGMENT_DENSITY_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::FRAGMENT_DENSITY_PROCESS_EXT,
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
    ];
    const RAYTRACING_PIPELINE_ORDER: [vk::PipelineStageFlags2; 2] = [
        vk::PipelineStageFlags2::DRAW_INDIRECT,
        vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
    ];
    const ALL_ORDERS: [&[vk::PipelineStageFlags2]; 5] = [
        &GRAPHICS_PIPELINE_ORDER,
        &GRAPHICS_MESH_PIPELINE_ORDER,
        &COMPUTE_PIPELINE_ORDER,
        &FRAGMENT_DENSITY_ORDER,
        &RAYTRACING_PIPELINE_ORDER,
    ];
    /// Compare two pipeline stages. Returns Some([`Ordering::Less`]) if `a` is earlier than `b`,
    /// [`Ordering::Equal`] if they are the same, [`Ordering::Greater`] if `a` is later than `b`,
    /// and [`None`] if they are not in the same time and are not mutually ordered.
    pub fn compare_pipeline_stages(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> Option<Ordering> {
        if a == b {
            return Some(std::cmp::Ordering::Equal);
        }
        for order in ALL_ORDERS.iter() {
            let first_index: Option<usize> = order.iter().position(|&x| a.contains(x));
            let second_index: Option<usize> = order.iter().position(|&x| b.contains(x));
            if let Some(first_index) = first_index
                && let Some(second_index) = second_index
            {
                return first_index.partial_cmp(&second_index);
            }
        }
        None
    }
    pub fn earlier_stage(
        a: vk::PipelineStageFlags2,
        b: vk::PipelineStageFlags2,
    ) -> vk::PipelineStageFlags2 {
        if let Some(ordering) = compare_pipeline_stages(a, b) {
            if ordering.is_le() { a } else { b }
        } else {
            a | b
        }
    }
}

#[cfg(test)]
mod tests {
    use super::utils::*;
    use super::*;
    #[test]
    fn test_earlier_stage() {
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::INDEX_INPUT,
                vk::PipelineStageFlags2::INDEX_INPUT
            ),
            vk::PipelineStageFlags2::INDEX_INPUT
        );
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::VERTEX_SHADER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            ),
            vk::PipelineStageFlags2::VERTEX_SHADER
        );
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            ),
            vk::PipelineStageFlags2::FRAGMENT_SHADER
        );
        assert_eq!(
            earlier_stage(
                vk::PipelineStageFlags2::VERTEX_SHADER,
                vk::PipelineStageFlags2::TRANSFER
            ),
            vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::TRANSFER
        );
    }
    #[test]
    fn test_compare_pipeline_stages() {
        assert!(
            compare_pipeline_stages(
                vk::PipelineStageFlags2::INDEX_INPUT,
                vk::PipelineStageFlags2::INDEX_INPUT
            )
            .unwrap()
            .is_eq()
        );
        assert!(
            compare_pipeline_stages(
                vk::PipelineStageFlags2::INDEX_INPUT,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            )
            .unwrap()
            .is_lt()
        );
        assert!(
            compare_pipeline_stages(
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            )
            .unwrap()
            .is_lt()
        );
        assert!(
            compare_pipeline_stages(
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
            )
            .unwrap()
            .is_lt()
        );
        assert!(
            compare_pipeline_stages(
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            )
            .unwrap()
            .is_gt()
        );
        assert!(
            compare_pipeline_stages(
                vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            )
            .is_none()
        );
        assert!(
            compare_pipeline_stages(
                vk::PipelineStageFlags2::TASK_SHADER_EXT,
                vk::PipelineStageFlags2::VERTEX_SHADER
            )
            .is_none()
        );
    }

    /// Write - Read - Read, reads are ordered-later
    #[test]
    fn test_wrr() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::empty()
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                    access: vk::AccessFlags2::empty()
                },
            }
        );
    }

    /// Write - Read - Read, reads are ordered-earlier
    #[test]
    fn test_wrr2() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        // The fragment shader resource usage would've been synced by the previous barrier too.
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
    }
    /// Write - Read - Read, reads are ordered the same
    #[test]
    fn test_wrr3() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        // The fragment shader resource usage would've been synced by the previous barrier too.
        assert_eq!(barrier, MemoryBarrier::default());
    }
    /// Write - Read - Read, reads are not ordered
    #[test]
    fn test_wrr4() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::TASK_SHADER_EXT,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::TASK_SHADER_EXT,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::MESH_SHADER_EXT,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
    }

    /// Write - Read - Read, layout transition between reads
    #[test]
    fn test_wrr5() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            true,
        );
        // The fragment shader resource usage would've been synced by the previous barrier too.
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::empty()
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
    }

    #[test]
    fn test_wrw() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                access: vk::AccessFlags2::SHADER_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                    access: vk::AccessFlags2::SHADER_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                    access: vk::AccessFlags2::empty()
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::empty()
                },
            }
        );
    }

    #[test]
    fn test_wrw2() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        // We need to wait on the read from the second operation to finish.
        // We also need to wait on the write from the first operation but this is done through an indirect barrier.
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::empty()
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::empty()
                },
            }
        );
    }

    #[test]
    fn test_www() {
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                access: vk::AccessFlags2::SHADER_WRITE,
            },
            false,
        );
        // We need to wait on the read from the second operation to finish.
        // We also need to wait on the write from the first operation but this is done through an indirect barrier.
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                    access: vk::AccessFlags2::SHADER_WRITE
                },
            }
        );
    }

    #[test]
    fn test_wrw_layout_transition() {
        // First draw samples a texture in the fragment shader. Second draw writes to that texture as a color attachment.
        let mut access = ResourceState::default();
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::TRANSFER,
                access: vk::AccessFlags2::TRANSFER_WRITE,
            },
            false,
        );
        assert_eq!(barrier, MemoryBarrier::default());
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                access: vk::AccessFlags2::SHADER_SAMPLED_READ,
            },
            false,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::TRANSFER,
                    access: vk::AccessFlags2::TRANSFER_WRITE
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::SHADER_SAMPLED_READ
                },
            }
        );
        let barrier = access.transition(
            Access {
                stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            },
            true,
        );
        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::empty()
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                },
            }
        );
    }

    #[test]
    fn test_wrr6() {
        let mut access = ResourceState::default();
        // Access 1: Transfer write
        let barrier = access.transition(Access::COPY_WRITE, false);
        assert_eq!(barrier, MemoryBarrier::default());

        // Access 2: Fragment shader read WITH layout transition
        let barrier = access.transition(Access::FRAGMENT_SAMPLED_READ, true);

        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access::COPY_WRITE,
                dst: Access::FRAGMENT_SAMPLED_READ,
            }
        );

        // Access 3: Vertex shader read, SAME layout (no transition)
        let barrier = access.transition(Access::VERTEX_SAMPLED_READ, false);

        assert_eq!(
            barrier,
            MemoryBarrier {
                src: Access {
                    stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    access: vk::AccessFlags2::empty()
                },
                dst: Access {
                    stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                    access: vk::AccessFlags2::empty()
                },
            }
        );
    }
}
