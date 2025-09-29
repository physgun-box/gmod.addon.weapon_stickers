local WeaponStickers = WeaponStickers or {}
WeaponStickers.Client = WeaponStickers.Client or {}

local client = WeaponStickers.Client
local cfg = WeaponStickers.Config
client.PlayerWeapons = client.PlayerWeapons or {}
client.WorldWeapons = client.WorldWeapons or {}
client.MaterialCache = client.MaterialCache or {}

local function readSticker()
    return {
        texture = net.ReadString() or "",
        bone = net.ReadString() or "",
        pos = net.ReadVector(),
        ang = net.ReadAngle(),
        size = net.ReadFloat()
    }
end

local function readStickerList()
    local count = net.ReadUInt(4)
    local list = {}

    for i = 1, count do
        list[i] = readSticker()
    end

    return list
end

local function writeSticker(sticker)
    net.WriteString(sticker.texture or "")
    net.WriteString(sticker.bone or "")
    net.WriteVector(sticker.pos or vector_origin)
    net.WriteAngle(sticker.ang or angle_zero)
    net.WriteFloat(math.Clamp(sticker.size or cfg.DefaultStickerSize, cfg.MinStickerSize, cfg.MaxStickerSize))
end

function client:GetPlayerStickerData()
    return self.PlayerWeapons
end

function client:GetWeaponStickers(weapon)
    if not IsValid(weapon) then return {} end

    local owner = weapon:GetOwner()

    if owner == LocalPlayer() then
        return self.PlayerWeapons[weapon:GetClass()] or {}
    end

    return self.WorldWeapons[weapon] or {}
end

function client:GetMaterial(texture)
    if texture == "" then return nil end

    if not self.MaterialCache[texture] then
        self.MaterialCache[texture] = Material(texture)
    end

    return self.MaterialCache[texture]
end

function client:RequestFullSync()
    net.Start("WeaponStickers_RequestData")
    net.SendToServer()
end

function client:SendStickerEdit(action, weaponClass, index, sticker)
    if not action or action == "" then return end
    if not weaponClass or weaponClass == "" then return end

    net.Start("WeaponStickers_Edit")
    net.WriteString(action)
    net.WriteString(weaponClass)

    if action == "add" then
        writeSticker(sticker or {})
    elseif action == "update" then
        net.WriteUInt(index or 0, 4)
        writeSticker(sticker or {})
    elseif action == "remove" then
        net.WriteUInt(index or 0, 4)
    elseif action == "clear" then
        -- nothing extra
    end

    net.SendToServer()
end

net.Receive("WeaponStickers_PlayerData", function()
    local weaponCount = net.ReadUInt(8)
    local data = {}

    for i = 1, weaponCount do
        local weaponClass = net.ReadString()
        data[weaponClass] = readStickerList()
    end

    client.PlayerWeapons = data

    hook.Run("WeaponStickers_PlayerDataUpdated", data)
end)

net.Receive("WeaponStickers_WeaponUpdate", function()
    local weapon = net.ReadEntity()
    local stickers = readStickerList()

    if not IsValid(weapon) or #stickers == 0 then
        client.WorldWeapons[weapon] = nil
    else
        client.WorldWeapons[weapon] = stickers
    end

    hook.Run("WeaponStickers_WeaponUpdated", weapon, stickers)
end)

hook.Add("InitPostEntity", "WeaponStickers_RequestInitial", function()
    timer.Simple(1, function()
        if not IsValid(LocalPlayer()) then return end
        client:RequestFullSync()
    end)
end)
