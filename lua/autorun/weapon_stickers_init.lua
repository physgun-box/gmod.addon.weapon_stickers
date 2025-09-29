WeaponStickers = WeaponStickers or {}

if SERVER then
    AddCSLuaFile("weapon_stickers/sh_config.lua")
    AddCSLuaFile("weapon_stickers/sh_data.lua")
    AddCSLuaFile("weapon_stickers/cl_net.lua")
    AddCSLuaFile("weapon_stickers/cl_render.lua")
    AddCSLuaFile("weapon_stickers/cl_gui.lua")

    include("weapon_stickers/sh_config.lua")
    include("weapon_stickers/sh_data.lua")
    include("weapon_stickers/sv_storage.lua")
    include("weapon_stickers/sv_net.lua")
else
    include("weapon_stickers/sh_config.lua")
    include("weapon_stickers/sh_data.lua")
    include("weapon_stickers/cl_net.lua")
    include("weapon_stickers/cl_render.lua")
    include("weapon_stickers/cl_gui.lua")
end
