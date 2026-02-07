.PHONY: default
default: triangle.spv

triangle.spv: triangle.slang
	slangc -O3 -target spirv -o triangle.spv triangle.slang
