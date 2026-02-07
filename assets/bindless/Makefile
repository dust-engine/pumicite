.PHONY: default
default: bindless.spv

bindless.spv: bindless.slang
	slangc -O3 -target spirv -o bindless.spv bindless.slang
