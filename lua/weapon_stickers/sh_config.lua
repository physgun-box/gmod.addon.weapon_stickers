WeaponStickers = WeaponStickers or {}
WeaponStickers.Config = WeaponStickers.Config or {}

local cfg = WeaponStickers.Config

cfg.MaxStickersPerWeapon = cfg.MaxStickersPerWeapon or 5
cfg.StorageDirectory = cfg.StorageDirectory or "weapon_stickers"
cfg.SaveDelay = cfg.SaveDelay or 2
cfg.DefaultStickerSize = cfg.DefaultStickerSize or 4
cfg.MinStickerSize = cfg.MinStickerSize or 1
cfg.MaxStickerSize = cfg.MaxStickerSize or 64
cfg.EditorCommand = cfg.EditorCommand or "ws_stickers"
