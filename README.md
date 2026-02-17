# Pumicite

A Vulkan rendering framework for Bevy that preserves direct, low-level GPU control
while using Bevy's ECS as the backbone for scheduling, concurrency, and resource
management.

<img src="https://repository-images.githubusercontent.com/1152040319/4887a04b-f01e-41ae-8aa4-5d9e2db84f48" height="250px">

## Why Pumicite?

Bevy's built-in renderer uses wgpu, which prioritizes safety over giving you direct
control of the GPU. Pumicite takes a different approach: instead of building a render
graph abstraction on top of the engine, it treats **Bevy systems as render graph nodes**
and **system ordering as node dependencies**. A schedule build pass automatically
handles command buffer allocation, barrier insertion, and queue submission.

You write Bevy systems that record Vulkan commands. The framework takes care of the
rest.

```rust
use bevy::prelude::*;
use bevy_pumicite::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);
    app.add_systems(PostUpdate, clear.in_set(DefaultRenderSet));
    app.run();
}

fn clear(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    time: Res<Time>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else { return };

    state.record(|encoder| {
        let Some(current) = swapchain_image.current_image() else { return };
        let current = encoder.lock(current, vk::PipelineStageFlags2::CLEAR);
        encoder.use_image_resource(
            current, &mut swapchain_image.state,
            Access::CLEAR, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1, 0..1, false,
        );
        encoder.emit_barriers();

        let hue = (time.elapsed_secs() * 72.0) % 360.0;
        let color: bevy::color::Srgba = bevy::color::Hsla::new(hue, 0.8, 0.5, 1.0).into();
        encoder.clear_color_image_with_layout(
            current,
            &vk::ClearColorValue { float32: color.to_f32_array() },
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
    });
}
```

## Key Features

- **System-as-Render-Graph** -- Bevy systems are render graph nodes. The ECS scheduler
  handles dependency tracking, mutual exclusion, and parallel execution. A
  `ScheduleBuildPass` transforms system sets into `vkQueueSubmit` calls.

- **Coroutine-as-Render-Graph** -- Record commands with Rust async/await, or ideally coroutines
  when it stabalizes. Yield points emit barriers and enable cross-future barrier merging.

- **GPUMutex** -- Timeline semaphore-based cross-queue synchronization. Lock a resource
  on a command encoder and semaphore waits are inserted automatically. Safe deferred
  cleanup via a recycler thread.

- **Resource State Tracking** -- Declare how you're about to use a resource and the
  framework computes the minimal pipeline barrier. You control the tracking granularity
  and where state is stored.

- **Bindless Rendering** -- Global descriptor heaps using `VK_EXT_mutable_descriptor_type`.
  All resources in one array, accessed by index through push constants. No descriptor
  set switching between draws. Forward compatible with `VK_EXT_descriptor_heap`.

- **Dynamic Rendering** -- No legacy `VkRenderPass` or `VkFramebuffer`. Attachments are
  specified inline when you begin rendering.

## Requirements

- **Rust**: Nightly
- **Vulkan**: 1.2+ with `VK_KHR_synchronization2` and `VK_KHR_timeline_semaphore`
- **Platform**: Windows, Linux, macOS (via MoltenVK or KosmicKrisp)

## Getting Started

### With Bevy

```toml
[dependencies]
bevy = { version = "0.17.0-dev", default-features = false, features = [
    "bevy_winit", "bevy_asset", "multi_threaded"
] }
bevy_pumicite = "0.1"

[patch.crates-io]
# Unfortunately, we need to fork bevy for now. The goal is to eventually upstream all the changes.
bevy = { git = "https://github.com/dust-engine/bevy", branch = "release-0.17.3" }
```

```rust
fn main() {
    App::new()
        .add_plugins(bevy_pumicite::DefaultPlugins)
        .run();
}
```

### Without Bevy

```toml
[dependencies]
pumicite = "0.1"
```

```rust
use pumicite::prelude::*;

let (device, mut queue) = Device::create_system_default().unwrap();
let allocator = Allocator::new(device.clone()).unwrap();
```

## Examples

Run examples with:

```bash
cargo run --example <name>
```

| Example | Description |
|---|---|
| `basics` | Headless image clear -- no window, no Bevy |
| `clear` | Window that clears to a cycling color |
| `triangle` | Graphics pipeline with dynamic rendering |
| `mandelbrot` | Interactive compute shader with push descriptors |
| `bindless` | Compute shader using bindless descriptor indexing |
| `sky_atmosphere` | Precomputed atmospheric scattering LUTs |
| `gltf` | glTF model loading with PBR shading |

## Crates

| Crate | Description |
|---|---|
| `pumicite` | Core Vulkan wrapper -- device, commands, sync, memory, pipelines |
| `bevy_pumicite` | Bevy integration -- plugins, submission sets, asset loaders, swapchain |
| `pumicite_egui` | egui integration for debug UIs |
| `pumicite_scene` | Scene and glTF loading |

## Tutorial

The [tutorial](https://github.com/dust-engine/pumicite/wiki) walks through Pumicite from the ground up:

1. [Overview](https://github.com/dust-engine/pumicite/wiki/Overview) -- Motivation, philosophy, and key concepts
2. [Getting Started](https://github.com/dust-engine/pumicite/wiki/Overview) -- Device creation and first command buffer
3. [Resource Management](https://github.com/dust-engine/pumicite/wiki/Resource%20Management) -- Buffers, images, and memory allocation
4. [Synchronization](https://github.com/dust-engine/pumicite/wiki/Syncronization) -- Barriers, resource state tracking, and GPUMutex
5. [Bevy Integration](https://github.com/dust-engine/pumicite/wiki/Bevy%20Integration) -- Plugins, submission sets, and the ECS render graph
6. [Compute](https://github.com/dust-engine/pumicite/wiki/Compute) -- Compute pipelines, dispatch, and multi-pass workflows
7. [Rendering](https://github.com/dust-engine/pumicite/wiki/Rendering) -- Dynamic rendering, graphics pipelines, and draw commands
8. [Bindless](https://github.com/dust-engine/pumicite/wiki/Bindless) -- Descriptor heaps and bindless resource indexing

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE).

---

**Note**: Pumicite is under active development. We offer no API stability guarantee until 1.0 release. Use at your own risk.
