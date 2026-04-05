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
    let num_writes = info.groups.len();

    // Collect struct fields, field defaults, new() descriptor_type assignments,
    // setter methods, and as_slice() pointer wiring.
    let mut struct_fields = Vec::new();
    let mut field_defaults = Vec::new();
    let mut new_assignments = Vec::new();
    let mut setter_methods = Vec::new();
    let mut as_slice_stmts = Vec::new();

    // Track cumulative offsets within each backing array for setters
    // in groups that have multiple entries.

    for (gi, group) in info.groups.iter().enumerate() {
        let gi_lit = proc_macro2::Literal::usize_unsuffixed(gi);
        let total = group.total_count() as usize;
        let dt_tokens = vk_descriptor_type_tokens(group.descriptor_type);
        let first_binding_lit = proc_macro2::Literal::u32_unsuffixed(group.first_binding);

        // new() assignment: set descriptor_type
        new_assignments.push(quote! {
            this.writes[#gi_lit].descriptor_type = #dt_tokens;
            this.writes[#gi_lit].dst_binding = #first_binding_lit;
        });

        match group.category {
            InfoCategory::AccelerationStructure => {
                let as_field = format_ident!("writes_as_{}", gi);
                let backing_field = format_ident!("writes_{}", gi);
                let total_lit = proc_macro2::Literal::usize_unsuffixed(total);

                struct_fields.push(quote! {
                    #as_field: pumicite::ash::vk::WriteDescriptorSetAccelerationStructureKHR<'static>,
                    #backing_field: [pumicite::ash::vk::AccelerationStructureKHR; #total_lit]
                });
                field_defaults.push(quote! {
                    #as_field: Default::default(),
                    #backing_field: Default::default()
                });

                as_slice_stmts.push(quote! {
                    self.writes[#gi_lit].p_next =
                        &self.#as_field as *const _ as *const core::ffi::c_void;
                    self.#as_field.acceleration_structure_count =
                        self.writes[#gi_lit].descriptor_count;
                    self.#as_field.p_acceleration_structures = self.#backing_field.as_ptr();
                });

                // Setter methods for each entry
                for entry in &group.entries {
                    let method = format_ident!("{}", entry.setter_name);
                    setter_methods.push(quote! {
                        pub fn #method(
                            &mut self,
                            accel: &impl pumicite::utils::AsVkHandle<Handle = pumicite::ash::vk::AccelerationStructureKHR>,
                        ) -> &mut Self {
                            let count = &mut self.writes[#gi_lit].descriptor_count;
                            self.#backing_field[*count as usize] =
                                pumicite::utils::AsVkHandle::vk_handle(accel);
                            *count += 1;
                            self
                        }
                    });
                }
            }
            InfoCategory::Buffer => {
                let backing_field = format_ident!("writes_{}", gi);
                let total_lit = proc_macro2::Literal::usize_unsuffixed(total);

                struct_fields.push(quote! {
                    #backing_field: [pumicite::ash::vk::DescriptorBufferInfo; #total_lit]
                });
                field_defaults.push(quote! {
                    #backing_field: Default::default()
                });

                as_slice_stmts.push(quote! {
                    self.writes[#gi_lit].p_buffer_info = self.#backing_field.as_ptr();
                });

                for entry in &group.entries {
                    let method = format_ident!("{}", entry.setter_name);
                    setter_methods.push(quote! {
                        pub fn #method(
                            &mut self,
                            buffer: &(impl pumicite::buffer::BufferLike + ?Sized),
                        ) -> &mut Self {
                            let count = &mut self.writes[#gi_lit].descriptor_count;
                            self.#backing_field[*count as usize] =
                                pumicite::ash::vk::DescriptorBufferInfo {
                                    buffer: pumicite::utils::AsVkHandle::vk_handle(buffer),
                                    offset: buffer.offset(),
                                    range: buffer.size(),
                                };
                            *count += 1;
                            self
                        }
                    });
                }
            }
            InfoCategory::Image => {
                let backing_field = format_ident!("writes_{}", gi);
                let total_lit = proc_macro2::Literal::usize_unsuffixed(total);

                struct_fields.push(quote! {
                    #backing_field: [pumicite::ash::vk::DescriptorImageInfo; #total_lit]
                });
                field_defaults.push(quote! {
                    #backing_field: Default::default()
                });

                as_slice_stmts.push(quote! {
                    self.writes[#gi_lit].p_image_info = self.#backing_field.as_ptr();
                });

                for entry in &group.entries {
                    let method = format_ident!("{}", entry.setter_name);

                    if entry.descriptor_type == DescriptorType::Sampler {
                        setter_methods.push(quote! {
                            pub fn #method(
                                &mut self,
                                sampler: &impl pumicite::utils::AsVkHandle<Handle = pumicite::ash::vk::Sampler>,
                            ) -> &mut Self {
                                let count = &mut self.writes[#gi_lit].descriptor_count;
                                self.#backing_field[*count as usize] =
                                    pumicite::ash::vk::DescriptorImageInfo {
                                        sampler: pumicite::utils::AsVkHandle::vk_handle(sampler),
                                        ..Default::default()
                                    };
                                *count += 1;
                                self
                            }
                        });
                    } else {
                        let layout = if is_storage_image(entry.descriptor_type) {
                            quote! { pumicite::ash::vk::ImageLayout::GENERAL }
                        } else {
                            quote! { pumicite::ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }
                        };

                        setter_methods.push(quote! {
                            pub fn #method(
                                &mut self,
                                image: &(impl pumicite::image::ImageViewLike + ?Sized),
                            ) -> &mut Self {
                                let count = &mut self.writes[#gi_lit].descriptor_count;
                                self.#backing_field[*count as usize] =
                                    pumicite::ash::vk::DescriptorImageInfo {
                                        image_view: pumicite::utils::AsVkHandle::vk_handle(image),
                                        image_layout: #layout,
                                        sampler: pumicite::ash::vk::Sampler::null(),
                                    };
                                *count += 1;
                                self
                            }
                        });
                    }
                }
            }
            InfoCategory::TexelBufferView => {
                let backing_field = format_ident!("writes_{}", gi);
                let total_lit = proc_macro2::Literal::usize_unsuffixed(total);

                struct_fields.push(quote! {
                    #backing_field: [pumicite::ash::vk::BufferView; #total_lit]
                });
                field_defaults.push(quote! {
                    #backing_field: Default::default()
                });

                as_slice_stmts.push(quote! {
                    self.writes[#gi_lit].p_texel_buffer_view = self.#backing_field.as_ptr();
                });

                for entry in &group.entries {
                    let method = format_ident!("{}", entry.setter_name);
                    setter_methods.push(quote! {
                        pub fn #method(
                            &mut self,
                            view: pumicite::ash::vk::BufferView,
                        ) -> &mut Self {
                            let count = &mut self.writes[#gi_lit].descriptor_count;
                            self.#backing_field[*count as usize] = view;
                            *count += 1;
                            self
                        }
                    });
                }
            }
        }
    }

    let num_writes_lit = proc_macro2::Literal::usize_unsuffixed(num_writes);

    quote! {
        #[derive(Clone)]
        pub struct #struct_name {
            writes: [pumicite::ash::vk::WriteDescriptorSet<'static>; #num_writes_lit],
            #(#struct_fields),*
        }

        impl #struct_name {
            pub fn new() -> Self {
                let mut this = Self {
                    writes: Default::default(),
                    #(#field_defaults),*
                };
                #(#new_assignments)*
                this
            }

            #(#setter_methods)*

            pub fn as_slice(&mut self) -> &[pumicite::ash::vk::WriteDescriptorSet] {
                #(#as_slice_stmts)*
                &self.writes
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
