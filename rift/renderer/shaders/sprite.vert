#version 460 core

// Static quad geometry (shared by every instance).
layout(location = 0) in vec2 in_position;   // quad corner, -0.5..0.5
layout(location = 1) in vec2 in_uv;         // quad corner uv, 0..1

// Per-instance attributes -- layout MUST match sprite_batch._INSTANCE_STRIDE.
layout(location = 2) in vec2 in_instance_pos;
layout(location = 3) in vec2 in_instance_scale;
layout(location = 4) in float in_instance_rotation;
layout(location = 5) in vec4 in_instance_uv_rect;      // xy = offset, zw = size, in atlas UV space
layout(location = 6) in uvec2 in_instance_tex_handle;  // bindless handle, low/high 32 bits

layout(location = 0) out vec2 out_uv;
layout(location = 1) out flat uvec2 out_tex_handle;

layout(std140, binding = 0) uniform CameraUBO {
    mat4 view_proj;
};

void main() {
    float c = cos(in_instance_rotation);
    float s = sin(in_instance_rotation);
    mat2 rotation = mat2(c, -s, s, c);
    vec2 world_pos = in_instance_pos + rotation * (in_position * in_instance_scale);

    gl_Position = view_proj * vec4(world_pos, 0.0, 1.0);
    out_uv = in_instance_uv_rect.xy + in_uv * in_instance_uv_rect.zw;
    out_tex_handle = in_instance_tex_handle;
}
