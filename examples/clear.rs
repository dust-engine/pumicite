use bevy::prelude::*;
use bevy_pumicite::{DefaultRenderSet, RenderState, swapchain::SwapchainImage};
use pumicite::prelude::*;
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
        .insert(bevy_pumicite::swapchain::SwapchainConfig {
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        });

    app.add_systems(PostUpdate, clear.in_set(DefaultRenderSet));
    app.run();
}

fn clear(
    mut swapchain_image: Query<&mut SwapchainImage, With<bevy::window::PrimaryWindow>>,
    mut state: RenderState,
    time: Res<Time>,
) {
    let Ok(mut swapchain_image) = swapchain_image.single_mut() else {
        return;
    };

    // Cycle through hues over time (full cycle every 5 seconds)
    let hue = (time.elapsed_secs() * 72.0) % 360.0;
    let color = bevy::color::Hsla::new(hue, 0.8, 0.5, 1.0);
    let rgba: bevy::color::Srgba = color.into();

    state.record(|encoder| {
        let Some(current_swapchain_image) = swapchain_image.current_image() else {
            return;
        };
        let current_swapchain_image =
            encoder.lock(current_swapchain_image, vk::PipelineStageFlags2::CLEAR);
        encoder.use_image_resource(
            current_swapchain_image,
            &mut swapchain_image.state,
            Access::CLEAR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        encoder.emit_barriers();
        encoder.clear_color_image(
            current_swapchain_image,
            &vk::ClearColorValue {
                float32: rgba.to_f32_array(),
            },
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
    });
}
