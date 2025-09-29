local WeaponStickers = WeaponStickers or {}
WeaponStickers.Data = WeaponStickers.Data or {}

local cfg = WeaponStickers.Config

function WeaponStickers.Data:SerialiseSticker(sticker)
    return {
        texture = tostring(sticker.texture or ""),
        bone = tostring(sticker.bone or ""),
        pos = sticker.pos and Vector(sticker.pos.x or 0, sticker.pos.y or 0, sticker.pos.z or 0) or vector_origin,
        ang = sticker.ang and Angle(sticker.ang.p or 0, sticker.ang.y or 0, sticker.ang.r or 0) or angle_zero,
        size = math.Clamp(tonumber(sticker.size) or cfg.DefaultStickerSize, cfg.MinStickerSize, cfg.MaxStickerSize)
    }
end

function WeaponStickers.Data:CopySticker(sticker)
    local s = self:SerialiseSticker(sticker)
    return {
        texture = s.texture,
        bone = s.bone,
        pos = Vector(s.pos.x, s.pos.y, s.pos.z),
        ang = Angle(s.ang.p, s.ang.y, s.ang.r),
        size = s.size
    }
end

function WeaponStickers.Data:NormaliseList(list)
    local normalised = {}

    if not istable(list) then
        return normalised
    end

    for _, sticker in ipairs(list) do
        normalised[#normalised + 1] = self:SerialiseSticker(sticker)
    end

    return normalised
end

function WeaponStickers.Data:IsValidSticker(sticker)
    if not istable(sticker) then return false end
    if sticker.texture == "" then return false end
    if sticker.size <= 0 then return false end
    return true
end

function WeaponStickers.Data:ClampList(list)
    list = list or {}
    if #list > cfg.MaxStickersPerWeapon then
        for i = #list, cfg.MaxStickersPerWeapon + 1, -1 do
            list[i] = nil
        end
    end

    return list
end

return WeaponStickers.Data
