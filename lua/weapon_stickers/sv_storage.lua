local WeaponStickers = WeaponStickers or {}
WeaponStickers.Storage = WeaponStickers.Storage or {}

local cfg = WeaponStickers.Config
local dataHelper = WeaponStickers.Data

local storage = WeaponStickers.Storage
storage.PlayerData = storage.PlayerData or {}
storage.PendingSaves = storage.PendingSaves or {}

local function getPlayerId(ply)
    if not IsValid(ply) then return nil end
    return ply:SteamID64() or ply:SteamID()
end

local function ensureDirectory()
    if not file.IsDir(cfg.StorageDirectory, "DATA") then
        file.CreateDir(cfg.StorageDirectory)
    end
end

local function getFilePath(ply)
    local id = getPlayerId(ply)
    if not id then return nil end
    return string.format("%s/%s.json", cfg.StorageDirectory, id)
end

function storage:GetPlayerData(ply)
    local id = getPlayerId(ply)
    if not id then return nil end

    self.PlayerData[id] = self.PlayerData[id] or {}
    return self.PlayerData[id]
end

function storage:LoadPlayer(ply)
    local path = getFilePath(ply)
    if not path then return end

    ensureDirectory()

    if not file.Exists(path, "DATA") then
        self.PlayerData[getPlayerId(ply)] = {}
        return
    end

    local json = file.Read(path, "DATA")
    local ok, decoded = pcall(util.JSONToTable, json or "{}")
    if not ok or not decoded then
        self.PlayerData[getPlayerId(ply)] = {}
        return
    end

    for weaponClass, stickers in pairs(decoded) do
        decoded[weaponClass] = dataHelper:NormaliseList(stickers)
    end

    self.PlayerData[getPlayerId(ply)] = decoded
end

function storage:ScheduleSave(ply)
    local id = getPlayerId(ply)
    if not id then return end

    if self.PendingSaves[id] then
        timer.Remove(self.PendingSaves[id])
    end

    local timerName = string.format("WeaponStickers_Save_%s", id)
    self.PendingSaves[id] = timerName

    timer.Create(timerName, cfg.SaveDelay, 1, function()
        self.PendingSaves[id] = nil
        self:SavePlayer(ply)
    end)
end

function storage:SavePlayer(ply)
    local path = getFilePath(ply)
    if not path then return end

    ensureDirectory()

    local data = self:GetPlayerData(ply)
    local serialised = {}

    for weaponClass, stickers in pairs(data or {}) do
        serialised[weaponClass] = {}

        for i, sticker in ipairs(stickers) do
            serialised[weaponClass][i] = {
                texture = sticker.texture,
                bone = sticker.bone,
                pos = { x = sticker.pos.x, y = sticker.pos.y, z = sticker.pos.z },
                ang = { p = sticker.ang.p, y = sticker.ang.y, r = sticker.ang.r },
                size = sticker.size
            }
        end
    end

    file.Write(path, util.TableToJSON(serialised, true))
end

function storage:GetWeaponStickers(ply, weaponClass)
    local data = self:GetPlayerData(ply)
    if not data then return {} end

    data[weaponClass] = dataHelper:ClampList(data[weaponClass])
    return data[weaponClass]
end

function storage:SetWeaponStickers(ply, weaponClass, stickers)
    local data = self:GetPlayerData(ply)
    if not data then return end

    data[weaponClass] = dataHelper:ClampList(dataHelper:NormaliseList(stickers))
    self:ScheduleSave(ply)
end

return storage
