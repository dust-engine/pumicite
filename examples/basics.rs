use ash::vk;
use pumicite::{command::CommandPool, prelude::*, sync::Timeline, utils::future::yield_now};

pub fn main() {
    let (device, mut queue) = Device::create_system_default().unwrap();
    let allocator = Allocator::new(device.clone()).unwrap();
    let mut timeline = Timeline::new(device.clone()).unwrap();

    let mut command_pool = CommandPool::new(device.clone(), 0).unwrap();
    let mut command_buffer = command_pool
        .alloc()
        .unwrap()
        .with_name(c"Demo Command Buffer");
    timeline.schedule(&mut command_buffer);
    command_pool.begin(&mut command_buffer).unwrap();

    let image_requirements = unsafe {
        let a = device
            .instance()
            .get_physical_device_image_format_properties(
                device.physical_device().vk_handle(),
                vk::Format::R8G8B8A8_UINT,
                vk::ImageType::TYPE_2D,
                vk::ImageTiling::LINEAR,
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::ImageCreateFlags::empty(),
            )
            .unwrap();
        println!("{:?}", a);
    };
    let image = Image::new_private(
        allocator,
        &vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format: vk::Format::R8G8B8A8_UINT,
            extent: vk::Extent3D {
                width: 128,
                height: 128,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::LINEAR,
            usage: vk::ImageUsageFlags::TRANSFER_DST,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        },
    )
    .unwrap();
    println!("{:?}", image_requirements);
    let image = GPUMutex::new(image);
    let mut resource_state = ResourceState::default();

    command_pool.record_future(&mut command_buffer, async |encoder| {
        let image = encoder.lock(&image, vk::PipelineStageFlags2::CLEAR);
        encoder.use_image_resource(
            image,
            &mut resource_state,
            Access::CLEAR,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            0..1,
            0..1,
            false,
        );
        yield_now().await;

        encoder.clear_color_image(
            &*image,
            &vk::ClearColorValue {
                uint32: [0, 0, 1, 2],
            },
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
    });

    command_pool.finish(&mut command_buffer).unwrap();
    queue.submit(&mut command_buffer).unwrap();
    command_buffer.block_until_completion().unwrap();
    command_pool.free(command_buffer);
    println!("Done!");
}
