use std::ops::Deref;
use std::sync::Arc;

use bevy::prelude::*;
use bevy_pumicite::prelude::*;
use bevy_reflect::TypePath;
use glam::UVec3;
use pumicite::bindless::ResourceHeap;
use pumicite::image::Image;
use pumicite::pipeline::Pipeline;
fn main() {
    let mut app = bevy::app::App::new();
    app.add_plugins(bevy_pumicite::DefaultPlugins);

    let primary_window = app
        .world_mut()
        .query_filtered::<Entity, With<bevy::window::PrimaryWindow>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut()
        .entity_mut(primary_window)
        .insert(SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST,
            ..Default::default()
        });

    app.add_systems(PostUpdate, clear.in_set(DefaultRenderSet));
    app.add_systems(Startup, setup);
    app.init_asset::<MyBindlessImageAsset>();
    app.enable_bindless().expect("Bindless not supported!");
    app.run();
}

#[derive(Asset, TypePath)]
struct MyBindlessImageAsset {
    heap: ResourceHeap,
    image: Arc<Image>,
    state: ResourceState,
    handle: u32,
}
impl Drop for MyBindlessImageAsset {
    fn drop(&mut self) {
        self.heap.remove(self.handle);
    }
}

#[derive(Resource)]
struct ExampleResource {
    pipeline: Handle<ComputePipeline>,
    image: Handle<MyBindlessImageAsset>,
}

fn setup(
    mut commands: Commands,
    asset_server: ResMut<AssetServer>,
    allocator: Res<Allocator>,
    heap: Res<DescriptorHeap>,
) {
    let allocator = allocator.clone();
    let heap = heap.resource_heap().clone();
    let image = asset_server.add_async(async move {
        let image = Image::new_private(
            allocator.clone(),
            &vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: 1024,
                    height: 1024,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                ..Default::default()
            },
        )?;
        Ok::<MyBindlessImageAsset, vk::Result>(MyBindlessImageAsset {
            handle: heap.add_image(&image, pumicite::bindless::ImageAccessMode::Storage)?,
            heap,
            image: Arc::new(image),
            state: ResourceState::default(),
        })
    });
    commands.insert_resource(ExampleResource {
        image,
        pipeline: asset_server.load("bindless/bindless.comp.pipeline.ron"),
    });
}

#[derive(bytemuck::Zeroable, bytemuck::NoUninit, Clone, Copy)]
#[repr(C)]
struct ExamplePushConstant {
    handle: u32,
    time: f32,
}

fn clear(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    example_resource: Res<ExampleResource>,
    pipelines: Res<Assets<ComputePipeline>>,
    mut example_assets: ResMut<Assets<MyBindlessImageAsset>>,
    time: Res<bevy::time::Time>,
    heap: Res<DescriptorHeap>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };
    let Some(pipeline) = pipelines.get(&example_resource.pipeline) else {
        return;
    };
    let Some(example) = example_assets.get_mut(&example_resource.image) else {
        return;
    };
    state.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };

        let target_image = encoder.retain(example.image.clone());

        encoder.use_image_resource(
            target_image.deref(),
            &mut example.state,
            Access::COMPUTE_WRITE,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            true,
        );
        encoder.emit_barriers();

        heap.bind(encoder, vk::PipelineBindPoint::COMPUTE);
        let pipeline: &Pipeline = encoder.retain(pipeline.clone().into_inner()).as_ref();
        encoder.bind_pipeline(vk::PipelineBindPoint::COMPUTE, pipeline);
        encoder.push_constants(
            pipeline.layout(),
            vk::ShaderStageFlags::ALL,
            0,
            &bytemuck::bytes_of(&ExamplePushConstant {
                handle: example.handle,
                time: time.elapsed_secs(),
            }),
        );
        encoder.dispatch(UVec3::new(128, 128, 1));

        let current_swapchain_image =
            encoder.lock(current_swapchain_image, vk::PipelineStageFlags2::CLEAR);
        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::BLIT_DST,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.use_image_resource(
            target_image.deref(),
            &mut example.state,
            Access::BLIT_SRC,
            vk::ImageLayout::GENERAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        encoder.blit_image(
            target_image.deref(),
            vk::ImageLayout::GENERAL,
            current_swapchain_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                src_offsets: [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: 1023,
                        y: 1023,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                dst_offsets: [
                    vk::Offset3D::default(),
                    vk::Offset3D {
                        x: current_swapchain_image.extent().x as i32 - 1,
                        y: current_swapchain_image.extent().y as i32 - 1,
                        z: 1,
                    },
                ],
            }],
            vk::Filter::NEAREST,
        );
    });
}
