# Pumicite

[![Crates.io](https://img.shields.io/crates/v/pumicite.svg)](https://crates.io/crates/pumicite)
[![Documentation](https://docs.rs/pumicite/badge.svg)](https://docs.rs/pumicite)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-2024%2B-orange.svg)](https://www.rust-lang.org)

A modern, high-performance Vulkan graphics library for Rust with first-class support for Bevy.

## Overview

Pumicite is a safe, ergonomic wrapper around Vulkan designed for high-performance graphics applications.
It targets modern desktop and mobile GPUs and seeks to enable graphics programmers to experiment and build
applications with the latest graphics features.

### Key Features

- **Bindless Heaps** - Full bindless support without pipeline / descriptor set layouts. Forward compatible with `VK_EXT_descriptor_heap`.
- **Automatic Resource Tracking** - Pipeline barriers and resource state management that strikes a balance
  between performance and usability.
- **Optimized Scheduling** - Parallel command buffer recording using Bevy system scheduler
- **Shading language agnostic** - Define pipeline states as bevy_assets, and consume them directly as a bevy asset.
  Use whatever shading language you prefer, as long as it compiles to SPIR-V.
- **Ray Tracing Support** - Hardware-accelerated ray tracing with `VK_KHR_ray_tracing_pipeline`
- **Memory Safety** - Safe abstractions over Vulkan with zero-cost guarantees
- **Async Command Recording** - Leverage Rust's Futures for preformant command buffer recording

## Who Should Use Pumicite?

**Perfect for:**
- Bevy developers who need more control than `bevy_render` and `wgpu` provides
- Vulkan developers seeking a safe, ergonomic Rust wrapper around Vulkan
- Rendering engineers building applications with cutting-edge GPU features
- Anyone needing direct, low-level access to Vulkan with safety guarantees

**Not suitable if:**
- You want a drop-in replacement for the default Bevy renderer
- `bevy_render` already meets your needs
- You're targeting web platforms (WebGPU)

## System Requirements

- **Rust**: Nightly
- **Vulkan**: 1.2+ with `VK_KHR_descriptor_indexing`, `VK_KHR_synchronization2` and `VK_KHR_timeline_semaphore`
- **Platform**: Windows, macOS (via [MoltenVK](https://github.com/KhronosGroup/MoltenVK) or [KosmicKrisp](https://docs.mesa3d.org/drivers/kosmickrisp.html)), or Linux

## Installation

Add Pumicite to your `Cargo.toml`:

```toml
[dependencies]
pumicite = "0.1.0"

# For Bevy integration
bevy_pumicite = "0.1.0"

# For egui support
pumicite_egui = "0.1.0"
```

### Feature Flags

```toml
[dependencies]
pumicite = { version = "0.1.0", features = ["bevy"] }
```

Available features:
- `bevy` - Enable Bevy ECS and asset system integration

## Quick Start

### Basic Vulkan Operations

```rust
use pumicite::{Device, command::{CommandPool, Timeline}};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create device and allocator
    let (device, mut queue) = Device::create_system_default()?;
    let mut timeline = Timeline::new(device.clone()).unwrap();
    
    // Create command pool and buffer
    let mut command_pool = CommandPool::new(device.clone(), 0)?;
    let mut command_buffer = command_pool.alloc()?;

    command_pool.begin(&mut command_buffer).unwrap();
    timeline.schedule(&mut command_buffer);
    
    // Record commands asynchronously
    command_pool.record_future(&mut command_buffer, async |encoder| {
        // Your rendering commands here
    });
    command_pool.finish(&mut command_buffer).unwrap();
    
    // Submit and wait
    queue.submit(&mut command_buffer)?;
    command_buffer.block_until_completion()?;
    
    Ok(())
}
```

### Bevy Integration

```rust
use bevy::prelude::*;
use bevy_pumicite::PumiciteApp;

fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);

    app.add_systems(PostUpdate, clear.in_set(DefaultRenderSet));
    app.run();
}
fn clear(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderSetSharedStateWrapper,
) {
    state.record(|encoder| {
        // Your render commands here
    });
}

```

## Examples

The repository includes comprehensive examples demonstrating various features:

### Core Examples
- **[`basics.rs`](examples/basics.rs)** - Fundamental Vulkan operations, resource management, and command recording
- **[`triangle.rs`](examples/triangle.rs)** - Classic triangle rendering with vertex buffers and graphics pipelines
- **[`clear.rs`](examples/clear.rs)** - Simple clear screen operations and swapchain management

### Advanced Examples
- **[`mandelbrot.rs`](examples/mandelbrot.rs)** - Compute shader example with interactive Mandelbrot set visualization
- **[`egui.rs`](examples/egui.rs)** - Immediate mode GUI integration with egui for debugging interfaces

## Architecture

Pumicite is built around several core concepts:

### System-as-render-graph
Bevy has a render graph at home! Render graph nodes are just a bevy systems, and node dependencies are defined as system orders.

A Bevy schedule build pass automatically optimize encoding order, merge render graph nodes, and call `vkQueueSubmit` on your behalf.

Other queue operations (like `vkQueueBindSparse`) are supported too!

```rust
impl Plugin for MyRenderPlugin {
    fn build(&self, app: &mut App) {
        app.add_device_extension::<KhrRayTracingPipeline>()
            .unwrap();
        app.enable_feature::<vk::PhysicalDeviceRayTracingPipelineFeatures>(|x| {
            &mut x.ray_tracing_pipeline
        })
        .unwrap();

        app.add_systems(
            bevy_app::PostUpdate, // Render systesm are executed in `PostUpdate` stage, maximizing parallelism with other systems
            (
                post_processing1, // Render graph nodes are systems
                post_processing2,
                gui_pass,
            ).chain().after(MainRenderSet),
        );
    }
}

```

### Async Command Recording (Experimental)
Commands can be recorded using Rust's async/await syntax. This allows the executor to merge pipeline barriers and optimize rendering orders across multiple
systems.

```rust
command_pool.record_future(&mut command_buffer, async |encoder| {
    let image = encoder.lock(&image, vk::PipelineStageFlags2::CLEAR);
    encoder.use_image_resource(image, &mut state, Access::CLEAR, layout, 0..1, 0..1, false);
    yield_now().await; // Cooperative yielding. The executor records all dependent work on the same submission, insert a pipeline barrier with dependency info, then resume encoding for subsequent work.
    encoder.clear_color_image(&*image, &clear_value);
});
```

### Resource Tracking
Automatic pipeline barrier insertion and resource state management. Resource states are optional and separate from
the resource itself, allowing you track as much or as little as you want to, at the granularity that works the best
for your specific use case - or forgo resource tracking alltogether for transient or read-only assets.

```rust
encoder.use_image_resource(
    image,
    &mut resource_state,
    Access::COMPUTE_WRITE,          // Access pattern with stage and access flags
    vk::ImageLayout::GENERAL,       // Target image layout
    0..1,                           // Mip level range
    0..1,                           // Array layer range
    false                           // Cannot discard original image content
);

// Emit the computed barriers
encoder.emit_barriers();

// Or, in an async context, the executor will batch barriers from multiple futures and emit them in one call to `vkCmdPipelineBarrier`.
yield_now().await;
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/pumicite.git
   cd pumicite
   ```

2. **Build and test**:
   ```bash
   cargo run --example triangle
   ```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your discretion.

---

**Note**: Pumicite is under active development. We offer no API stability guarnatee until 1.0 release.
