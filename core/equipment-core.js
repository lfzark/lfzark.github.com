(function attachEquipmentCore(global) {
  "use strict";

  function clampInt(v, min, max) {
    const n = Number.isFinite(v) ? Math.floor(v) : min;
    return Math.max(min, Math.min(max, n));
  }

  function getRarityMultiplier(rarity) {
    if (rarity === "epic") return 2.05;
    if (rarity === "rare") return 1.62;
    if (rarity === "uncommon") return 1.28;
    return 1;
  }

  function calcEnhanceCost(rarity, level) {
    const lv = clampInt(level, 0, 10);
    const rarityMul = getRarityMultiplier(rarity);
    return {
      gold: Math.floor((30 + lv * 20 + lv * lv * 3) * rarityMul),
      spiritStone: Math.max(6, Math.floor((8 + lv * 6) * rarityMul)),
      demonCore: rarity === "epic" ? Math.max(1, Math.floor((lv + 1) / 2)) : lv >= 4 ? 1 : 0,
      rareCrystal: rarity === "epic" ? Math.max(1, Math.floor((lv + 2) / 3)) : rarity === "rare" && lv >= 6 ? 1 : 0
    };
  }

  function createEquipInstance(itemId, location, enhance, uidNum) {
    return {
      uid: "eq_" + clampInt(uidNum, 1, Number.MAX_SAFE_INTEGER),
      itemId,
      location,
      enhance: clampInt(enhance, 0, 10)
    };
  }

  function groupInstancesByLevel(instances, itemId) {
    const map = new Map();
    instances.forEach((it) => {
      if (!it || it.itemId !== itemId) return;
      const lv = clampInt(it.enhance || 0, 0, 10);
      map.set(lv, (map.get(lv) || 0) + 1);
    });
    return map;
  }

  global.SimpleWorldEquipCore = {
    calcEnhanceCost,
    createEquipInstance,
    groupInstancesByLevel
  };
})(window);

