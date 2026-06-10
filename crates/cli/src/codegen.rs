use pumicite_types::DescriptorType;
use quote::{format_ident, quote};
use shader_slang as slang;

// ---------------------------------------------------------------------------
// Intermediate types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InfoCategory {
    Buffer,
    Image,
    TexelBufferView,
    AccelerationStructure,
}

#[derive(Clone)]
struct FlatBinding {
    setter_name: String,
    descriptor_type: DescriptorType,
    category: InfoCategory,
    count: u32,
}

struct WriteGroup {
    descriptor_type: DescriptorType,
    category: InfoCategory,
    first_binding: u32,
    entries: Vec<FlatBinding>,
}

impl WriteGroup {
    fn total_count(&self) -> u32 {
        self.entries.iter().map(|e| e.count).sum()
    }
}

struct ParameterBlockInfo {
    struct_name: String,
    groups: Vec<WriteGroup>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn info_category(dt: DescriptorType) -> InfoCategory {
    match dt {
        DescriptorType::UniformBuffer
        | DescriptorType::StorageBuffer
        | DescriptorType::UniformBufferDynamic
        | DescriptorType::StorageBufferDynamic => InfoCategory::Buffer,

        DescriptorType::Sampler
        | DescriptorType::CombinedImageSampler
        | DescriptorType::SampledImage
        | DescriptorType::StorageImage
        | DescriptorType::InputAttachment => InfoCategory::Image,

        DescriptorType::UniformTexelBuffer | DescriptorType::StorageTexelBuffer => {
            InfoCategory::TexelBufferView
        }

        DescriptorType::AccelerationStructure => InfoCategory::AccelerationStructure,

        _ => panic!("unsupported descriptor type for codegen"),
    }
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                // Don't insert underscore between consecutive uppercase (e.g. "BVH" -> "bvh")
                let prev = s.as_bytes()[i - 1] as char;
                if prev.is_lowercase() || prev.is_ascii_digit() {
                    result.push('_');
                } else if i + 1 < s.len() {
                    let next = s.as_bytes()[i + 1] as char;
                    if next.is_lowercase() {
                        result.push('_');
                    }
                }
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
}

fn vk_descriptor_type_tokens(dt: DescriptorType) -> proc_macro2::TokenStream {
    match dt {
        DescriptorType::Sampler => quote! { pumicite::ash::vk::DescriptorType::SAMPLER },
        DescriptorType::CombinedImageSampler => {
            quote! { pumicite::ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER }
        }
        DescriptorType::SampledImage => quote! { pumicite::ash::vk::DescriptorType::SAMPLED_IMAGE },
        DescriptorType::StorageImage => quote! { pumicite::ash::vk::DescriptorType::STORAGE_IMAGE },
        DescriptorType::UniformTexelBuffer => {
            quote! { pumicite::ash::vk::DescriptorType::UNIFORM_TEXEL_BUFFER }
        }
        DescriptorType::StorageTexelBuffer => {
            quote! { pumicite::ash::vk::DescriptorType::STORAGE_TEXEL_BUFFER }
        }
        DescriptorType::UniformBuffer => {
            quote! { pumicite::ash::vk::DescriptorType::UNIFORM_BUFFER }
        }
        DescriptorType::StorageBuffer => {
            quote! { pumicite::ash::vk::DescriptorType::STORAGE_BUFFER }
        }
        DescriptorType::UniformBufferDynamic => {
            quote! { pumicite::ash::vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC }
        }
        DescriptorType::StorageBufferDynamic => {
            quote! { pumicite::ash::vk::DescriptorType::STORAGE_BUFFER_DYNAMIC }
        }
        DescriptorType::InputAttachment => {
            quote! { pumicite::ash::vk::DescriptorType::INPUT_ATTACHMENT }
        }
        DescriptorType::AccelerationStructure => {
            quote! { pumicite::ash::vk::DescriptorType::ACCELERATION_STRUCTURE_KHR }
        }
        _ => panic!("unsupported descriptor type"),
    }
}

fn is_storage_image(dt: DescriptorType) -> bool {
    matches!(dt, DescriptorType::StorageImage)
}

// ---------------------------------------------------------------------------
// Reflection walking
// ---------------------------------------------------------------------------

fn flatten_bindings(
    parent_tl: &slang::reflection::TypeLayout,
    prefix: &str,
    out: &mut Vec<FlatBinding>,
) {
    let field_count = parent_tl.field_count();
    let total_ranges = parent_tl.binding_range_count();

    for fi in 0..field_count {
        let field = parent_tl.field_by_index(fi).unwrap();
        let field_name = field.name().unwrap_or("unknown");
        let field_tl = field.type_layout().unwrap();

        let setter_name = if prefix.is_empty() {
            to_snake_case(field_name)
        } else {
            format!("{}_{}", prefix, to_snake_case(field_name))
        };

        // Check if this field is a struct (recurse) or a leaf binding
        let kind = field_tl.kind();
        if kind == slang::TypeKind::Struct {
            flatten_bindings(field_tl, &setter_name, out);
            continue;
        }

        // Leaf field: get binding ranges from the parent's field_binding_range_offset
        let range_start = parent_tl.field_binding_range_offset(fi as i64);
        let range_end = if fi + 1 < field_count {
            parent_tl.field_binding_range_offset((fi + 1) as i64)
        } else {
            total_ranges
        };

        for ri in range_start..range_end {
            let binding_type = parent_tl.binding_range_type(ri);
            let Some(descriptor_type) = crate::slang_binding_type_to_descriptor_type(binding_type)
            else {
                continue;
            };
            let count = parent_tl.binding_range_binding_count(ri).max(1) as u32;
            let category = info_category(descriptor_type);

            // Use leaf variable name if available, otherwise use field name
            let name = parent_tl
                .binding_range_leaf_variable(ri)
                .and_then(|v| v.name())
                .map(|n| {
                    if prefix.is_empty() {
                        to_snake_case(n)
                    } else {
                        format!("{}_{}", prefix, to_snake_case(n))
                    }
                })
                .unwrap_or_else(|| setter_name.clone());

            out.push(FlatBinding {
                setter_name: name,
                descriptor_type,
                category,
                count,
            });
        }
    }
}

fn collapse_bindings(bindings: Vec<FlatBinding>) -> Vec<WriteGroup> {
    let mut groups: Vec<WriteGroup> = Vec::new();
    let mut current_binding: u32 = 0;

    for binding in bindings {
        let should_merge = if let Some(last) = groups.last() {
            last.descriptor_type == binding.descriptor_type
                && binding.count == 1
                && last.entries.iter().all(|e| e.count == 1)
        } else {
            false
        };

        if should_merge {
            groups.last_mut().unwrap().entries.push(binding.clone());
        } else {
            groups.push(WriteGroup {
                descriptor_type: binding.descriptor_type,
                category: binding.category,
                first_binding: current_binding,
                entries: vec![binding.clone()],
            });
        }
        current_binding += binding.count;
    }

    groups
}

fn discover_parameter_blocks(reflection: &slang::reflection::Shader) -> Vec<ParameterBlockInfo> {
    let mut result = Vec::new();

    for param in reflection.parameters() {
        let Some(tl) = param.type_layout() else {
            continue;
        };
        if tl.kind() != slang::TypeKind::ParameterBlock {
            continue;
        }

        let element_tl = match tl.element_type_layout() {
            Some(el) => el,
            None => continue,
        };

        let struct_name = element_tl
            .name()
            .unwrap_or("UnknownParameterBlock")
            .to_string();

        let mut flat = Vec::new();
        flatten_bindings(element_tl, "", &mut flat);

        if flat.is_empty() {
            continue;
        }

        // Deduplicate names
        let mut seen = std::collections::HashMap::<String, u32>::new();
        for binding in &mut flat {
            let count = seen.entry(binding.setter_name.clone()).or_insert(0);
            *count += 1;
            if *count > 1 {
                binding.setter_name = format!("{}_{}", binding.setter_name, count);
            }
        }

        let groups = collapse_bindings(flat);

        result.push(ParameterBlockInfo {
            struct_name,
            groups,
        });
    }

    result
}

// ---------------------------------------------------------------------------
// Code generation
// ---------------------------------------------------------------------------

fn generate_struct(info: &ParameterBlockInfo) -> proc_macro2::TokenStream {
    let struct_name = format_ident!("{}", info.struct_name);

    // Worst case one WriteDescriptorSet per descriptor element (alternating
    // dirty/clean pattern); sized at codegen time so as_slice() never allocates.
    let max_writes: usize = info.groups.iter().map(|g| g.total_count() as usize).sum();
    let max_writes_lit = proc_macro2::Literal::usize_unsuffixed(max_writes);

    let mut struct_fields = Vec::new();
    let mut field_defaults = Vec::new();
    let mut setter_methods = Vec::new();
    let mut as_slice_blocks = Vec::new();

    for (gi, group) in info.groups.iter().enumerate() {
        let total = group.total_count() as usize;
        assert!(
            total <= 64,
            "parameter block `{}`: write group at binding {} has {} descriptors, \
             but dirty tracking supports at most 64 per group",
            info.struct_name,
            group.first_binding,
            total
        );

        let backing_field = format_ident!("writes_{}", gi);
        let dirty_field = format_ident!("dirty_{}", gi);
        let dt_tokens = vk_descriptor_type_tokens(group.descriptor_type);
        let first_binding_lit = proc_macro2::Literal::u32_unsuffixed(group.first_binding);
        let total_lit = proc_macro2::Literal::usize_unsuffixed(total);

        // A group is either consecutive single-descriptor bindings (a dirty run
        // offsets the binding index) or one arrayed binding (a dirty run offsets
        // the array element); collapse_bindings never mixes the two.
        let is_arrayed = group.entries.len() == 1 && group.entries[0].count > 1;
        let dst_tokens = if is_arrayed {
            quote! {
                dst_binding: #first_binding_lit,
                dst_array_element: start,
            }
        } else {
            quote! {
                dst_binding: #first_binding_lit + start,
            }
        };

        let info_ty = match group.category {
            InfoCategory::Buffer => quote! { pumicite::ash::vk::DescriptorBufferInfo },
            InfoCategory::Image => quote! { pumicite::ash::vk::DescriptorImageInfo },
            InfoCategory::TexelBufferView => quote! { pumicite::ash::vk::BufferView },
            InfoCategory::AccelerationStructure => {
                quote! { pumicite::ash::vk::AccelerationStructureKHR }
            }
        };

        struct_fields.push(quote! {
            #backing_field: [#info_ty; #total_lit],
            #dirty_field: u64
        });
        field_defaults.push(quote! {
            #backing_field: [#info_ty::default(); #total_lit],
            #dirty_field: 0
        });

        let (run_prelude, run_body) = match group.category {
            InfoCategory::AccelerationStructure => {
                let as_field = format_ident!("writes_as_{}", gi);
                struct_fields.push(quote! {
                    #as_field: [pumicite::ash::vk::WriteDescriptorSetAccelerationStructureKHR<'static>; #total_lit]
                });
                field_defaults.push(quote! {
                    #as_field: [pumicite::ash::vk::WriteDescriptorSetAccelerationStructureKHR::default(); #total_lit]
                });
                (
                    quote! { let mut r: usize = 0; },
                    quote! {
                        self.#as_field[r] = pumicite::ash::vk::WriteDescriptorSetAccelerationStructureKHR {
                            acceleration_structure_count: len,
                            p_acceleration_structures: &self.#backing_field[start as usize],
                            ..Default::default()
                        };
                        self.writes[n] = pumicite::ash::vk::WriteDescriptorSet {
                            #dst_tokens
                            descriptor_count: len,
                            descriptor_type: #dt_tokens,
                            p_next: &self.#as_field[r] as *const _ as *const core::ffi::c_void,
                            ..Default::default()
                        };
                        r += 1;
                    },
                )
            }
            _ => {
                let info_assign = match group.category {
                    InfoCategory::Buffer => {
                        quote! { p_buffer_info: &self.#backing_field[start as usize], }
                    }
                    InfoCategory::Image => {
                        quote! { p_image_info: &self.#backing_field[start as usize], }
                    }
                    InfoCategory::TexelBufferView => {
                        quote! { p_texel_buffer_view: &self.#backing_field[start as usize], }
                    }
                    InfoCategory::AccelerationStructure => unreachable!(),
                };
                (
                    quote! {},
                    quote! {
                        self.writes[n] = pumicite::ash::vk::WriteDescriptorSet {
                            #dst_tokens
                            descriptor_count: len,
                            descriptor_type: #dt_tokens,
                            #info_assign
                            ..Default::default()
                        };
                    },
                )
            }
        };

        as_slice_blocks.push(quote! {
            {
                #run_prelude
                let mut mask = self.#dirty_field;
                self.#dirty_field = 0;
                while mask != 0 {
                    let start = mask.trailing_zeros();
                    let len = (!(mask >> start)).trailing_zeros();
                    mask &= !((u64::MAX >> (64 - len)) << start);
                    #run_body
                    n += 1;
                }
            }
        });

        let mut write_i: u32 = 0;
        for entry in &group.entries {
            let method = format_ident!("{}", entry.setter_name);

            // Arrayed bindings take a runtime element index; single bindings
            // address their fixed slot within the group's backing array.
            let (index_param, slot_tokens, bit_tokens, guard_tokens) = if entry.count > 1 {
                let count_lit = proc_macro2::Literal::u32_unsuffixed(entry.count);
                (
                    quote! { index: u32, },
                    quote! { index as usize },
                    quote! { index },
                    quote! { debug_assert!(index < #count_lit); },
                )
            } else {
                let slot_lit = proc_macro2::Literal::usize_unsuffixed(write_i as usize);
                let bit_lit = proc_macro2::Literal::u32_unsuffixed(write_i);
                (
                    quote! {},
                    quote! { #slot_lit },
                    quote! { #bit_lit },
                    quote! {},
                )
            };

            let (value_param, value_expr) = match group.category {
                InfoCategory::AccelerationStructure => (
                    quote! { accel: &impl pumicite::utils::AsVkHandle<Handle = pumicite::ash::vk::AccelerationStructureKHR> },
                    quote! { pumicite::utils::AsVkHandle::vk_handle(accel) },
                ),
                InfoCategory::Buffer => (
                    quote! { buffer: &(impl pumicite::buffer::BufferLike + ?Sized) },
                    quote! {
                        pumicite::ash::vk::DescriptorBufferInfo {
                            buffer: pumicite::utils::AsVkHandle::vk_handle(buffer),
                            offset: buffer.offset(),
                            range: buffer.size(),
                        }
                    },
                ),
                InfoCategory::Image => {
                    if entry.descriptor_type == DescriptorType::Sampler {
                        (
                            quote! { sampler: &impl pumicite::utils::AsVkHandle<Handle = pumicite::ash::vk::Sampler> },
                            quote! {
                                pumicite::ash::vk::DescriptorImageInfo {
                                    sampler: pumicite::utils::AsVkHandle::vk_handle(sampler),
                                    ..Default::default()
                                }
                            },
                        )
                    } else {
                        let layout = if is_storage_image(entry.descriptor_type) {
                            quote! { pumicite::ash::vk::ImageLayout::GENERAL }
                        } else {
                            quote! { pumicite::ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }
                        };
                        (
                            quote! { image: &(impl pumicite::image::ImageViewLike + ?Sized) },
                            quote! {
                                pumicite::ash::vk::DescriptorImageInfo {
                                    image_view: pumicite::utils::AsVkHandle::vk_handle(image),
                                    image_layout: #layout,
                                    sampler: pumicite::ash::vk::Sampler::null(),
                                }
                            },
                        )
                    }
                }
                InfoCategory::TexelBufferView => (
                    quote! { view: pumicite::ash::vk::BufferView },
                    quote! { view },
                ),
            };

            setter_methods.push(quote! {
                pub fn #method(&mut self, #index_param #value_param) -> &mut Self {
                    #guard_tokens
                    self.#backing_field[#slot_tokens] = #value_expr;
                    self.#dirty_field |= 1u64 << #bit_tokens;
                    self
                }
            });

            write_i += entry.count;
        }
    }

    quote! {
        #[derive(Clone)]
        pub struct #struct_name {
            writes: [pumicite::ash::vk::WriteDescriptorSet<'static>; #max_writes_lit],
            #(#struct_fields),*
        }

        impl #struct_name {
            pub fn new() -> Self {
                Self {
                    writes: [pumicite::ash::vk::WriteDescriptorSet::default(); #max_writes_lit],
                    #(#field_defaults),*
                }
            }

            #(#setter_methods)*

            /// Returns descriptor writes covering only the bindings modified since
            /// the last call, coalesced into contiguous runs. Consumes the
            /// modified flags: a subsequent call returns an empty slice until a
            /// setter is called again.
            pub fn as_slice(&mut self) -> &[pumicite::ash::vk::WriteDescriptorSet<'_>] {
                let mut n: usize = 0;
                #(#as_slice_blocks)*
                &self.writes[..n]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn generate(reflection: &slang::reflection::Shader) -> String {
    let blocks = discover_parameter_blocks(reflection);
    let mut tokens = proc_macro2::TokenStream::new();
    for block in &blocks {
        tokens.extend(generate_struct(block));
    }
    tokens.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(name: &str, dt: DescriptorType, count: u32) -> FlatBinding {
        FlatBinding {
            setter_name: name.to_string(),
            descriptor_type: dt,
            category: info_category(dt),
            count,
        }
    }

    // The output of this test is the source for the behavioral snapshot test in
    // //tests/codegen_generated.rs (pumicite root crate). Regenerate with
    // `cargo test -p pumicite_cli -- --nocapture` after changing codegen.
    #[test]
    fn generate_synthetic_block() {
        let flat = vec![
            binding("uniforms", DescriptorType::UniformBuffer, 1),
            binding("nodes", DescriptorType::StorageBuffer, 1),
            binding("leaves", DescriptorType::StorageBuffer, 1),
            binding("albedo", DescriptorType::SampledImage, 1),
            binding("normal", DescriptorType::SampledImage, 1),
            binding("depth", DescriptorType::StorageImage, 1),
            binding("linear_sampler", DescriptorType::Sampler, 1),
            binding("cascades", DescriptorType::SampledImage, 4),
            binding("scene_bvh", DescriptorType::AccelerationStructure, 1),
        ];
        let groups = collapse_bindings(flat);

        // uniform | storage x2 | sampled x2 | storage image | sampler |
        // sampled[4] | acceleration structure
        assert_eq!(groups.len(), 7);
        assert_eq!(groups[1].first_binding, 1);
        assert_eq!(groups[1].entries.len(), 2);
        assert_eq!(groups[2].first_binding, 3);
        assert_eq!(groups[5].first_binding, 7);
        assert_eq!(groups[6].first_binding, 11);

        let info = ParameterBlockInfo {
            struct_name: "TestParams".to_string(),
            groups,
        };
        let code = generate_struct(&info).to_string();
        println!("{code}");

        // Arrayed binding generates an indexed setter and array-element runs.
        assert!(code.contains("dst_array_element"));
        assert!(code.contains("index : u32"));
    }
}
