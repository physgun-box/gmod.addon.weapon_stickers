local WeaponStickers = WeaponStickers or {}
local client = WeaponStickers.Client
local cfg = WeaponStickers.Config

local function gatherWeapons()
    local weaponMap = {}
    local ply = LocalPlayer()

    if IsValid(ply) then
        for _, weapon in ipairs(ply:GetWeapons()) do
            if IsValid(weapon) then
                weaponMap[weapon:GetClass()] = weapon:GetPrintName() ~= "" and weapon:GetPrintName() or weapon:GetClass()
            end
        end
    end

    for weaponClass in pairs(client.PlayerWeapons or {}) do
        weaponMap[weaponClass] = weaponMap[weaponClass] or weaponClass
    end

    local list = {}
    for weaponClass, name in pairs(weaponMap) do
        list[#list + 1] = { class = weaponClass, name = name }
    end

    table.sort(list, function(a, b)
        return string.lower(a.name) < string.lower(b.name)
    end)

    return list
end

local function createNumSlider(form, label, min, max, decimals, default)
    local slider = form:NumSlider(label, nil, min, max)
    slider:SetDecimals(decimals or 2)
    slider:SetValue(default or 0)
    return slider
end

local function populateFormFields(fields, sticker)
    sticker = sticker or {
        texture = "",
        bone = "",
        pos = vector_origin,
        ang = angle_zero,
        size = cfg.DefaultStickerSize
    }

    fields.texture:SetValue(sticker.texture or "")
    fields.bone:SetValue(sticker.bone or "")

    fields.posX:SetValue(sticker.pos and sticker.pos.x or 0)
    fields.posY:SetValue(sticker.pos and sticker.pos.y or 0)
    fields.posZ:SetValue(sticker.pos and sticker.pos.z or 0)

    fields.angP:SetValue(sticker.ang and sticker.ang.p or 0)
    fields.angY:SetValue(sticker.ang and sticker.ang.y or 0)
    fields.angR:SetValue(sticker.ang and sticker.ang.r or 0)

    fields.size:SetValue(sticker.size or cfg.DefaultStickerSize)
end

local function collectFormFields(fields)
    return {
        texture = fields.texture:GetValue(),
        bone = fields.bone:GetValue(),
        pos = Vector(fields.posX:GetValue(), fields.posY:GetValue(), fields.posZ:GetValue()),
        ang = Angle(fields.angP:GetValue(), fields.angY:GetValue(), fields.angR:GetValue()),
        size = fields.size:GetValue()
    }
end

local function refreshStickerList(stickerList, weaponClass)
    stickerList:Clear()
    if not weaponClass or weaponClass == "" then return end

    local stickers = client.PlayerWeapons[weaponClass] or {}

    for i, sticker in ipairs(stickers) do
        local bone = sticker.bone ~= "" and sticker.bone or "-"
        stickerList:AddLine(i, sticker.texture, bone, string.format("%.2f", sticker.size or 0))
    end
end

local function ensureWeaponChoice(combo, selectedClass)
    combo:Clear()

    local weapons = gatherWeapons()
    local defaultClass = selectedClass
    if not defaultClass and IsValid(LocalPlayer()) then
        local active = LocalPlayer():GetActiveWeapon()
        if IsValid(active) then
            defaultClass = active:GetClass()
        end
    end

    local foundDefault = false

    for _, data in ipairs(weapons) do
        local text = string.format("%s (%s)", data.name, data.class)
        combo:AddChoice(text, data.class, data.class == defaultClass)
        if data.class == defaultClass then
            foundDefault = true
        end
    end

    if not foundDefault and weapons[1] then
        combo:ChooseOptionID(1)
        return weapons[1].class
    end

    return defaultClass
end

local function showLimitNotification()
    notification.AddLegacy(string.format("Достигнут предел в %d стикеров", cfg.MaxStickersPerWeapon), NOTIFY_ERROR, 3)
    surface.PlaySound("buttons/button10.wav")
end

local function openEditor()
    if not IsValid(LocalPlayer()) then return end

    local frame = vgui.Create("DFrame")
    frame:SetSize(960, 600)
    frame:SetTitle("Стикеры на оружие")
    frame:Center()
    frame:MakePopup()

    local hookId = "WeaponStickers_GUI_" .. frame:GetCreationID()

    local topPanel = vgui.Create("DPanel", frame)
    topPanel:Dock(TOP)
    topPanel:SetTall(36)
    topPanel:DockMargin(0, 4, 0, 4)
    topPanel:SetPaintBackground(false)

    local weaponCombo = vgui.Create("DComboBox", topPanel)
    weaponCombo:Dock(LEFT)
    weaponCombo:SetWide(320)
    weaponCombo:SetSortItems(false)

    local refreshButton = vgui.Create("DButton", topPanel)
    refreshButton:Dock(RIGHT)
    refreshButton:SetWide(120)
    refreshButton:SetText("Обновить")

    local infoLabel = vgui.Create("DLabel", topPanel)
    infoLabel:Dock(FILL)
    infoLabel:SetText("Выберите оружие для редактирования")
    infoLabel:SetContentAlignment(4)

    local main = vgui.Create("DPanel", frame)
    main:Dock(FILL)
    main:DockPadding(8, 8, 8, 8)
    main:SetPaintBackground(false)

    local leftPanel = vgui.Create("DPanel", main)
    leftPanel:Dock(LEFT)
    leftPanel:SetWide(380)
    leftPanel:DockPadding(0, 0, 8, 0)
    leftPanel:SetPaintBackground(false)

    local stickerList = vgui.Create("DListView", leftPanel)
    stickerList:Dock(FILL)
    stickerList:SetMultiSelect(false)
    stickerList:AddColumn("#"):SetFixedWidth(30)
    stickerList:AddColumn("Материал")
    stickerList:AddColumn("Кость")
    stickerList:AddColumn("Размер")

    local buttonPanel = vgui.Create("DPanel", leftPanel)
    buttonPanel:Dock(BOTTOM)
    buttonPanel:SetTall(120)
    buttonPanel:DockMargin(0, 8, 0, 0)
    buttonPanel:SetPaintBackground(false)

    local addButton = vgui.Create("DButton", buttonPanel)
    addButton:Dock(TOP)
    addButton:DockMargin(0, 0, 0, 4)
    addButton:SetText("Добавить стикер")

    local applyButton = vgui.Create("DButton", buttonPanel)
    applyButton:Dock(TOP)
    applyButton:DockMargin(0, 0, 0, 4)
    applyButton:SetText("Применить изменения")

    local removeButton = vgui.Create("DButton", buttonPanel)
    removeButton:Dock(TOP)
    removeButton:DockMargin(0, 0, 0, 4)
    removeButton:SetText("Удалить выбранный")

    local clearButton = vgui.Create("DButton", buttonPanel)
    clearButton:Dock(TOP)
    clearButton:SetText("Очистить все")

    local rightPanel = vgui.Create("DPanel", main)
    rightPanel:Dock(FILL)
    rightPanel:SetPaintBackground(false)

    local form = vgui.Create("DForm", rightPanel)
    form:Dock(FILL)
    form:SetName("Параметры стикера")

    form:Help("Введите путь к материалу и параметры позиционирования.")
    form:Help("Используйте квадратные текстуры для лучшего результата.")

    local fields = {}
    fields.texture = form:TextEntry("Материал", "")
    fields.bone = form:TextEntry("Кость", "")

    form:Help("Локальная позиция относительно выбранной кости (в единицах модели).")
    fields.posX = createNumSlider(form, "Позиция X", -50, 50, 2, 0)
    fields.posY = createNumSlider(form, "Позиция Y", -50, 50, 2, 0)
    fields.posZ = createNumSlider(form, "Позиция Z", -50, 50, 2, 0)

    form:Help("Поворот стикера (Pitch/Yaw/Roll).")
    fields.angP = createNumSlider(form, "Pitch", -180, 180, 1, 0)
    fields.angY = createNumSlider(form, "Yaw", -180, 180, 1, 0)
    fields.angR = createNumSlider(form, "Roll", -180, 180, 1, 0)

    form:Help(string.format("Размер наклейки (ограничение %d).", cfg.MaxStickerSize))
    fields.size = createNumSlider(form, "Размер", cfg.MinStickerSize, cfg.MaxStickerSize, 2, cfg.DefaultStickerSize)

    local selectedWeapon
    local selectedIndex

    local function updateInfoLabel()
        if not selectedWeapon then
            infoLabel:SetText("Выберите оружие для редактирования")
            return
        end

        local stickers = client.PlayerWeapons[selectedWeapon] or {}
        infoLabel:SetText(string.format("%s — %d/%d стикеров", selectedWeapon, #stickers, cfg.MaxStickersPerWeapon))
    end

    local function refreshAll()
        selectedWeapon = ensureWeaponChoice(weaponCombo, selectedWeapon)
        refreshStickerList(stickerList, selectedWeapon)
        updateInfoLabel()
    end

    refreshAll()

    weaponCombo.OnSelect = function(_, _, _, data)
        selectedWeapon = data
        selectedIndex = nil
        populateFormFields(fields, nil)
        refreshStickerList(stickerList, selectedWeapon)
        updateInfoLabel()
    end

    refreshButton.DoClick = function()
        client:RequestFullSync()
        refreshAll()
    end

    stickerList.OnRowSelected = function(panel, rowIndex, row)
        selectedIndex = row:GetValue(1)
        local stickers = client.PlayerWeapons[selectedWeapon] or {}
        populateFormFields(fields, stickers[tonumber(selectedIndex)] or nil)
    end

    addButton.DoClick = function()
        if not selectedWeapon then return end
        local stickers = client.PlayerWeapons[selectedWeapon] or {}
        if #stickers >= cfg.MaxStickersPerWeapon then
            showLimitNotification()
            return
        end

        client:SendStickerEdit("add", selectedWeapon, nil, collectFormFields(fields))
        timer.Simple(0.1, refreshAll)
    end

    applyButton.DoClick = function()
        if not selectedWeapon or not selectedIndex then return end
        client:SendStickerEdit("update", selectedWeapon, tonumber(selectedIndex) or 1, collectFormFields(fields))
        timer.Simple(0.1, refreshAll)
    end

    removeButton.DoClick = function()
        if not selectedWeapon or not selectedIndex then return end
        client:SendStickerEdit("remove", selectedWeapon, tonumber(selectedIndex) or 1)
        timer.Simple(0.1, function()
            selectedIndex = nil
            populateFormFields(fields, nil)
            refreshAll()
        end)
    end

    clearButton.DoClick = function()
        if not selectedWeapon then return end
        client:SendStickerEdit("clear", selectedWeapon)
        timer.Simple(0.1, function()
            selectedIndex = nil
            populateFormFields(fields, nil)
            refreshAll()
        end)
    end

    local function onDataUpdated()
        if not IsValid(frame) then
            hook.Remove("WeaponStickers_PlayerDataUpdated", hookId)
            return
        end

        refreshAll()
    end

    hook.Add("WeaponStickers_PlayerDataUpdated", hookId, onDataUpdated)

    frame.OnClose = function()
        hook.Remove("WeaponStickers_PlayerDataUpdated", hookId)
    end

    populateFormFields(fields, nil)
end

concommand.Add(cfg.EditorCommand, function()
    openEditor()
end)

hook.Add("OnPlayerChat", "WeaponStickers_ChatCommand", function(ply, text)
    if ply ~= LocalPlayer() then return end
    text = string.Trim(string.lower(text or ""))
    if text == "!stickers" or text == "!стикеры" then
        openEditor()
        return true
    end
end)
