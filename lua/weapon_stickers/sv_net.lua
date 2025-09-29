local WeaponStickers = WeaponStickers or {}
local storage = WeaponStickers.Storage
local dataHelper = WeaponStickers.Data
local cfg = WeaponStickers.Config

util.AddNetworkString("WeaponStickers_PlayerData")
util.AddNetworkString("WeaponStickers_WeaponUpdate")
util.AddNetworkString("WeaponStickers_Edit")
util.AddNetworkString("WeaponStickers_RequestData")

local function writeSticker(sticker)
    net.WriteString(sticker.texture or "")
    net.WriteString(sticker.bone or "")
    net.WriteVector(sticker.pos or vector_origin)
    net.WriteAngle(sticker.ang or angle_zero)
    net.WriteFloat(sticker.size or cfg.DefaultStickerSize)
end

local function writeStickerList(list)
    list = list or {}
    net.WriteUInt(#list, 4)
    for _, sticker in ipairs(list) do
        writeSticker(sticker)
    end
end

local function readSticker()
    local texture = net.ReadString() or ""
    local bone = net.ReadString() or ""
    local pos = net.ReadVector()
    local ang = net.ReadAngle()
    local size = net.ReadFloat()

    return dataHelper:SerialiseSticker({
        texture = texture,
        bone = bone,
        pos = pos,
        ang = ang,
        size = size
    })
end

local function sendPlayerData(ply)
    if not IsValid(ply) then return end

    local data = storage:GetPlayerData(ply) or {}
    net.Start("WeaponStickers_PlayerData")
    net.WriteUInt(table.Count(data), 8)

    for weaponClass, stickers in pairs(data) do
        net.WriteString(weaponClass)
        writeStickerList(stickers)
    end

    net.Send(ply)
end

local function sendWeaponUpdate(weapon, stickers, target)
    if not IsValid(weapon) then return end

    net.Start("WeaponStickers_WeaponUpdate")
    net.WriteEntity(weapon)
    writeStickerList(stickers)

    if target then
        if istable(target) then
            net.Send(target)
        else
            net.Send(target)
        end
    else
        net.Broadcast()
    end
end

function WeaponStickers.SyncWeapon(ply, weapon, target)
    if not IsValid(ply) or not IsValid(weapon) then return end

    local data = storage:GetWeaponStickers(ply, weapon:GetClass())
    sendWeaponUpdate(weapon, data, target)
end

function WeaponStickers.SyncPlayerWeapons(ply, target)
    if not IsValid(ply) then return end

    for _, weapon in ipairs(ply:GetWeapons()) do
        if IsValid(weapon) then
            WeaponStickers.SyncWeapon(ply, weapon, target)
        end
    end
end

local function handleStickerEdit(len, ply)
    if not IsValid(ply) then return end

    local action = net.ReadString() or ""
    local weaponClass = net.ReadString() or ""
    if weaponClass == "" then return end

    local weapon = ply:GetWeapon(weaponClass)
    if not IsValid(weapon) then return end

    local playerData = storage:GetPlayerData(ply)
    if not playerData then return end

    playerData[weaponClass] = playerData[weaponClass] or {}
    local list = playerData[weaponClass]

    local changed = false

    if action == "add" then
        if #list >= cfg.MaxStickersPerWeapon then return end
        local sticker = readSticker()
        if not dataHelper:IsValidSticker(sticker) then return end
        list[#list + 1] = sticker
        changed = true
    elseif action == "update" then
        local index = net.ReadUInt(4)
        if not list[index] then return end
        local sticker = readSticker()
        if not dataHelper:IsValidSticker(sticker) then return end
        list[index] = sticker
        changed = true
    elseif action == "remove" then
        local index = net.ReadUInt(4)
        if not list[index] then return end
        table.remove(list, index)
        changed = true
    elseif action == "clear" then
        if #list == 0 then return end
        for i = #list, 1, -1 do
            list[i] = nil
        end
        changed = true
    end

    if not changed then return end

    storage:SetWeaponStickers(ply, weaponClass, list)
    sendPlayerData(ply)

    local updatedList = storage:GetWeaponStickers(ply, weaponClass)
    local activeWeapon = ply:GetWeapon(weaponClass)
    if IsValid(activeWeapon) then
        sendWeaponUpdate(activeWeapon, updatedList)
    end
end

net.Receive("WeaponStickers_Edit", handleStickerEdit)

net.Receive("WeaponStickers_RequestData", function(_, ply)
    sendPlayerData(ply)
end)

hook.Add("PlayerInitialSpawn", "WeaponStickers_LoadData", function(ply)
    storage:LoadPlayer(ply)

    timer.Simple(1, function()
        if not IsValid(ply) then return end
        sendPlayerData(ply)
        for _, other in ipairs(player.GetAll()) do
            if IsValid(other) then
                WeaponStickers.SyncPlayerWeapons(other, ply)
            end
        end
    end)
end)

hook.Add("PlayerDisconnected", "WeaponStickers_SaveData", function(ply)
    storage:SavePlayer(ply)
end)

hook.Add("PlayerSpawn", "WeaponStickers_SyncOnSpawn", function(ply)
    timer.Simple(0, function()
        if not IsValid(ply) then return end
        WeaponStickers.SyncPlayerWeapons(ply)
    end)
end)

hook.Add("PlayerSwitchWeapon", "WeaponStickers_SyncOnSwitch", function(ply, oldWep, newWep)
    if not IsValid(ply) then return end
    timer.Simple(0, function()
        if IsValid(newWep) then
            WeaponStickers.SyncWeapon(ply, newWep)
        end
    end)
end)

hook.Add("EntityRemoved", "WeaponStickers_ClearRemovedWeapon", function(ent)
    if not ent:IsWeapon() then return end

    net.Start("WeaponStickers_WeaponUpdate")
    net.WriteEntity(ent)
    net.WriteUInt(0, 4)
    net.Broadcast()
end)
