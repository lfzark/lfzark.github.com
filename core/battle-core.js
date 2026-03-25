(function attachBattleCore(global) {
  "use strict";

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function calcLootBase(ctx) {
    const tier = ctx.monsterTier || 1;
    const tierMul = tier >= 5 ? 3.2 : tier >= 4 ? 2.4 : tier >= 3 ? 1.6 : 1;
    const eventDropMul = 1 + (ctx.eventDropMul || 0) + (ctx.arrayDropMul || 0);
    const rewardBoost = 1.6;
    const gainStone = Math.floor((8 + ctx.monsterLevel * 4) * (1 + ctx.riskBonus * 0.9) * tierMul * eventDropMul * rewardBoost);
    const gainHerb = ctx.monsterLevel >= 2 ? Math.max(1, Math.floor((ctx.riskBonus >= 0.35 ? 2 : 1) * tierMul * rewardBoost)) : 0;
    const rareChance = clamp((ctx.rareBase || 12) + (ctx.fortune || 0) + Math.floor((ctx.luck || 0) * 0.8), 0, ctx.rareMax || 45);
    const coreBase = tier >= 3 ? 24 : tier === 2 ? 14 : 8;
    const coreChance = clamp(coreBase + Math.floor((ctx.fortune || 0) * 0.6), 0, 55);
    return { tierMul, eventDropMul, rewardBoost, gainStone, gainHerb, rareChance, coreChance };
  }

  global.SimpleWorldBattleCore = {
    calcLootBase
  };
})(window);

