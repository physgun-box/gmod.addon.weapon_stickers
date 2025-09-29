local WeaponStickers = WeaponStickers or {}
local client = WeaponStickers.Client
local cfg = WeaponStickers.Config

client.WhiteMaterial = client.WhiteMaterial or Material("vgui/white")

local function buildMatrixFromBone(ent, boneName)
    if not IsValid(ent) then return nil end

    local boneId

    if boneName ~= "" then
        boneId = ent:LookupBone(boneName)
    end

    if not boneId or boneId < 0 then
        return nil
    end

    local matrix = ent:GetBoneMatrix(boneId)

    if matrix then
        return matrix
    end

    local pos, ang = ent:GetBonePosition(boneId)
    if not pos or not ang then
        return nil
    end

    matrix = Matrix()
    matrix:SetTranslation(pos)
    matrix:SetAngles(ang)
    return matrix
end

local function drawSticker(ent, sticker)
    if not IsValid(ent) then return end

    local matrix = buildMatrixFromBone(ent, sticker.bone or "")
    local basePos
    local baseAng

    if matrix then
        basePos = matrix:GetTranslation()
        baseAng = matrix:GetAngles()
    else
        basePos = ent:GetPos()
        baseAng = ent:GetAngles()
    end

    local pos = LocalToWorld(sticker.pos or vector_origin, angle_zero, basePos, baseAng)
    local ang = LocalToWorldAngles(sticker.ang or angle_zero, baseAng)

    local offset = ang:Forward() * 0.1
    pos = pos + offset

    local scale = math.Clamp(sticker.size or cfg.DefaultStickerSize, cfg.MinStickerSize, cfg.MaxStickerSize)

    cam.Start3D2D(pos, ang, 1)
        surface.SetDrawColor(255, 255, 255, 255)
        local material = client:GetMaterial(sticker.texture)
        surface.SetMaterial(material or client.WhiteMaterial)
        surface.DrawTexturedRect(-scale * 0.5, -scale * 0.5, scale, scale)
    cam.End3D2D()
end

local function drawStickers(ent, stickers)
    if not IsValid(ent) then return end
    if not stickers or #stickers == 0 then return end

    render.SuppressEngineLighting(true)
    for _, sticker in ipairs(stickers) do
        drawSticker(ent, sticker)
    end
    render.SuppressEngineLighting(false)
end

hook.Add("PostDrawViewModel", "WeaponStickers_DrawViewModel", function(vm, ply, weapon)
    if not IsValid(vm) or not IsValid(ply) then return end
    if ply ~= LocalPlayer() then return end
    if not IsValid(weapon) then return end

    local stickers = client:GetPlayerStickerData()[weapon:GetClass()]
    drawStickers(vm, stickers)
end)

hook.Add("PostDrawTranslucentRenderables", "WeaponStickers_DrawWorld", function()
    for weapon, stickers in pairs(client.WorldWeapons) do
        if IsValid(weapon) then
            drawStickers(weapon, stickers)
        else
            client.WorldWeapons[weapon] = nil
        end
    end
end)
