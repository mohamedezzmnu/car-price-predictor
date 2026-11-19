/* ==========================================================================
   CarPulse — Client-Side Linear Regression Engine & Liquid Glass UI
   ========================================================================== */

const REFERENCE_YEAR = 2026;

const MANUFACTURERS = [
  "ACURA", "ALFA ROMEO", "ASTON MARTIN", "AUDI", "BMW", "BUICK", "CADILLAC",
  "CHEVROLET", "CHRYSLER", "CITROEN", "DAEWOO", "DAIHATSU", "DODGE", "FERRARI",
  "FIAT", "FORD", "GAZ", "GMC", "GREATWALL", "HAVAL", "HONDA", "HUMMER",
  "HYUNDAI", "INFINITI", "ISUZU", "JAGUAR", "JEEP", "KIA", "LANCIA",
  "LAND ROVER", "LEXUS", "LINCOLN", "MASERATI", "MAZDA", "MERCEDES-BENZ",
  "MERCURY", "MINI", "MITSUBISHI", "NISSAN", "OPEL", "PEUGEOT", "PONTIAC",
  "PORSCHE", "RENAULT", "ROVER", "SAAB", "SATURN", "SCION", "SEAT", "SKODA",
  "SSANGYONG", "SUBARU", "SUZUKI", "TESLA", "TOYOTA", "UAZ", "VAZ",
  "VOLKSWAGEN", "VOLVO", "ZAZ", "სხვა"
];

const CATEGORIES = [
  "Cabriolet", "Coupe", "Goods wagon", "Hatchback", "Jeep",
  "Limousine", "Microbus", "Minivan", "Pickup", "Sedan", "Universal"
];

const FUEL_TYPES = [
  "CNG", "Diesel", "Hybrid", "Hydrogen", "LPG", "Petrol", "Plug-in Hybrid"
];

const GEAR_BOXES = [
  "Automatic", "Manual", "Tiptronic", "Variator"
];

const PRESETS = {
  toyota: {
    Manufacturer: "TOYOTA",
    Category: "Sedan",
    ProdYear: 2022,
    Doors: 4,
    EngineVolume: 2.5,
    Cylinders: 4,
    FuelType: "Petrol",
    GearBox: "Tiptronic",
    DriveWheels: "Front",
    Turbo: false,
    Mileage: 65000,
    Levy: 779,
    Airbags: 8,
    Leather: true,
    Wheel: false
  },
  hyundai: {
    Manufacturer: "HYUNDAI",
    Category: "Sedan",
    ProdYear: 2021,
    Doors: 4,
    EngineVolume: 2.0,
    Cylinders: 4,
    FuelType: "Petrol",
    GearBox: "Automatic",
    DriveWheels: "Front",
    Turbo: false,
    Mileage: 82000,
    Levy: 640,
    Airbags: 6,
    Leather: true,
    Wheel: false
  },
  bmw: {
    Manufacturer: "BMW",
    Category: "Jeep",
    ProdYear: 2023,
    Doors: 4,
    EngineVolume: 4.8,
    Cylinders: 8,
    FuelType: "Petrol",
    GearBox: "Tiptronic",
    DriveWheels: "4x4",
    Turbo: true,
    Mileage: 48000,
    Levy: 1250,
    Airbags: 12,
    Leather: true,
    Wheel: false
  },
  mercedes: {
    Manufacturer: "MERCEDES-BENZ",
    Category: "Sedan",
    ProdYear: 2020,
    Doors: 4,
    EngineVolume: 2.7,
    Cylinders: 6,
    FuelType: "Diesel",
    GearBox: "Tiptronic",
    DriveWheels: "Rear",
    Turbo: true,
    Mileage: 110000,
    Levy: 890,
    Airbags: 10,
    Leather: true,
    Wheel: false
  },
  tesla: {
    Manufacturer: "TESLA",
    Category: "Sedan",
    ProdYear: 2024,
    Doors: 4,
    EngineVolume: 2.0,
    Cylinders: 4,
    FuelType: "Plug-in Hybrid",
    GearBox: "Automatic",
    DriveWheels: "4x4",
    Turbo: false,
    Mileage: 24000,
    Levy: 0,
    Airbags: 12,
    Leather: true,
    Wheel: false
  }
};

let MODEL = null;
let currentState = { ...PRESETS.toyota };
let compareSlotA = { ...PRESETS.toyota };
let compareSlotB = { ...PRESETS.bmw };
let nextCompareSlot = "B";
let displayedPrice = 0;
let animFrameId = null;

// ==========================================================================
// 1. Initialization
// ==========================================================================
document.addEventListener("DOMContentLoaded", async () => {
  populateDropdowns();
  initSpecularGlassEffect();
  bindUIEvents();

  try {
    const res = await fetch("model.json");
    MODEL = await res.json();
    applyStateToForm(currentState);
    updatePrediction();
    renderComparison();
  } catch (err) {
    console.error("Failed to load model.json:", err);
    showToast("تعذر تحميل ملف معاملات النموذج");
  }
});

function populateDropdowns() {
  fillSelect("manufacturer", MANUFACTURERS, (val) =>
    val === "სხვა" ? "Other (أخرى)" : val
  );
  fillSelect("category", CATEGORIES);
  fillSelect("fuelType", FUEL_TYPES);
  fillSelect("gearBox", GEAR_BOXES);
}

function fillSelect(id, items, labelFn = (v) => v) {
  const el = document.getElementById(id);
  el.innerHTML = items
    .map((item) => `<option value="${item}">${labelFn(item)}</option>`)
    .join("");
}

// ==========================================================================
// 2. Linear Regression Inference Engine (Calibrated to 2026)
// ==========================================================================
function predictCarPrice(state) {
  if (!MODEL) return null;

  const { intercept, coef, mean, scale, sigma } = MODEL;

  // Relative car age in 2026, bounded to the model's trained [0..40] age domain
  const carAge = Math.max(0, Math.min(40, REFERENCE_YEAR - Number(state.ProdYear)));
  const logMileage = Math.log1p(Math.max(1, Number(state.Mileage)));

  const rawNumeric = {
    Levy: Number(state.Levy),
    "Engine volume": Number(state.EngineVolume),
    Mileage: logMileage,
    Cylinders: Number(state.Cylinders),
    Airbags: Number(state.Airbags),
    "Car age": carAge,
    Doors: Number(state.Doors)
  };

  let logPred = intercept;
  const contributions = [];

  for (const col of MODEL.num_cols) {
    const z = (rawNumeric[col] - mean[col]) / scale[col];
    const delta = z * (coef[col] || 0);
    logPred += delta;
    contributions.push({
      key: col,
      label: formatFeatureLabel(col, state, carAge),
      delta
    });
  }

  const binaryMap = {
    "Leather interior": state.Leather ? 1 : 0,
    Wheel: state.Wheel ? 1 : 0,
    Turbo: state.Turbo ? 1 : 0
  };

  for (const [col, val] of Object.entries(binaryMap)) {
    const delta = val * (coef[col] || 0);
    logPred += delta;
    if (val === 1 || Math.abs(delta) > 0.001) {
      contributions.push({
        key: col,
        label: formatFeatureLabel(col, state, carAge),
        delta
      });
    }
  }

  const oneHotSpecs = [
    { prefix: "Manufacturer_", val: state.Manufacturer, label: `الماركة: ${state.Manufacturer}` },
    { prefix: "Category_", val: state.Category, label: `الفئة: ${state.Category}` },
    { prefix: "Fuel type_", val: state.FuelType, label: `الوقود: ${state.FuelType}` },
    { prefix: "Gear box type_", val: state.GearBox, label: `ناقل الحركة: ${state.GearBox}` },
    { prefix: "Drive wheels_", val: state.DriveWheels, label: `الدفع: ${state.DriveWheels}` }
  ];

  for (const spec of oneHotSpecs) {
    const featureKey = `${spec.prefix}${spec.val}`;
    const delta = coef[featureKey] !== undefined ? coef[featureKey] : 0;
    logPred += delta;
    contributions.push({
      key: spec.prefix,
      label: spec.label,
      delta
    });
  }

  const price = Math.max(2000, Math.expm1(logPred));
  const priceLow = Math.max(1500, Math.expm1(logPred - sigma));
  const priceHigh = Math.expm1(logPred + sigma);

  contributions.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  return {
    price,
    priceLow,
    priceHigh,
    logPred,
    carAge,
    contributions
  };
}

function formatFeatureLabel(col, state, carAge) {
  switch (col) {
    case "Car age":
      return carAge === 0 ? "سنة الصنع (جديد 2026)" : `عمر السيارة (${carAge} سنة)`;
    case "Engine volume":
      return `سعة المحرك (${Number(state.EngineVolume).toFixed(1)}L)`;
    case "Mileage":
      return `المسافة (${Number(state.Mileage).toLocaleString("en-US")} km)`;
    case "Turbo":
      return state.Turbo ? "شاحن تيربو (Turbo)" : "بدون تيربو";
    case "Airbags":
      return `الوسائد الهوائية (${state.Airbags})`;
    case "Wheel":
      return state.Wheel ? "مقود يمين (RHD)" : "مقود يسار";
    case "Levy":
      return `الرسوم الجمركية ($${state.Levy})`;
    case "Leather interior":
      return state.Leather ? "فرش جلد" : "فرش قماش";
    case "Doors":
      return `عدد الأبواب (${state.Doors})`;
    case "Cylinders":
      return `الأسطوانات (${state.Cylinders} سلندر)`;
    default:
      return col;
  }
}

// ==========================================================================
// 3. UI Rendering
// ==========================================================================
function updatePrediction() {
  const result = predictCarPrice(currentState);
  if (!result) return;

  animatePriceCounter(Math.round(result.price));

  document.getElementById("logPredVal").textContent = `ln(y) = ${result.logPred.toFixed(3)}`;
  document.getElementById("priceLow").textContent = `$${Math.round(result.priceLow).toLocaleString("en-US")}`;
  document.getElementById("priceHigh").textContent = `$${Math.round(result.priceHigh).toLocaleString("en-US")}`;

  const pillsContainer = document.getElementById("carSummaryPills");
  const tags = [
    `${currentState.Manufacturer} • ${currentState.ProdYear}`,
    `${currentState.Category}`,
    `${Number(currentState.EngineVolume).toFixed(1)}L ${currentState.Turbo ? "Turbo" : ""}`.trim(),
    `${currentState.FuelType}`,
    `${currentState.GearBox}`,
    `${currentState.DriveWheels}`
  ];
  pillsContainer.innerHTML = tags.map((t) => `<span class="spec-pill">${t}</span>`).join("");

  renderContributions(result.contributions.slice(0, 5));
}

function animatePriceCounter(target) {
  if (animFrameId) cancelAnimationFrame(animFrameId);
  const start = displayedPrice || target * 0.88;
  const duration = 320;
  const startTime = performance.now();
  const el = document.getElementById("predictedPrice");

  function step(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const val = Math.round(start + (target - start) * eased);
    displayedPrice = val;
    el.textContent = val.toLocaleString("en-US");
    if (progress < 1) {
      animFrameId = requestAnimationFrame(step);
    }
  }
  animFrameId = requestAnimationFrame(step);
}

function renderContributions(items) {
  const container = document.getElementById("contributionsList");
  const maxAbs = Math.max(...items.map((i) => Math.abs(i.delta)), 0.25);

  container.innerHTML = items
    .map((item) => {
      const isPos = item.delta >= 0;
      const pct = Math.min(100, Math.round((Math.abs(item.delta) / maxAbs) * 100));
      const pctImpact = ((Math.exp(item.delta) - 1) * 100).toFixed(1);
      const sign = item.delta >= 0 ? "+" : "";
      return `
        <div class="contrib-item">
          <div class="contrib-top">
            <span class="contrib-name">${item.label}</span>
            <span class="contrib-val ${isPos ? "pos" : "neg"}">${sign}${pctImpact}%</span>
          </div>
          <div class="contrib-track">
            <div class="contrib-bar ${isPos ? "pos" : "neg"}" style="width: ${pct}%"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

// ==========================================================================
// 4. Form Synchronization & Events
// ==========================================================================
function applyStateToForm(state) {
  document.getElementById("manufacturer").value = state.Manufacturer;
  document.getElementById("category").value = state.Category;
  document.getElementById("prodYear").value = state.ProdYear;
  document.getElementById("engineVolume").value = state.EngineVolume;
  document.getElementById("cylinders").value = state.Cylinders;
  document.getElementById("fuelType").value = state.FuelType;
  document.getElementById("gearBox").value = state.GearBox;
  document.getElementById("mileage").value = state.Mileage;
  document.getElementById("levy").value = state.Levy;
  document.getElementById("airbags").value = state.Airbags;
  document.getElementById("turbo").checked = Boolean(state.Turbo);
  document.getElementById("leather").checked = Boolean(state.Leather);
  document.getElementById("wheel").checked = Boolean(state.Wheel);

  document.querySelectorAll(".segmented-control").forEach((group) => {
    const key = group.dataset.name;
    group.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.classList.toggle("active", String(btn.dataset.value) === String(state[key]));
    });
  });

  syncSliderLabels();
}

function syncSliderLabels() {
  const year = Number(currentState.ProdYear);
  const age = REFERENCE_YEAR - year;
  const ageLabel = age === 0 ? "موديل السنة" : `عمر: ${age} سنة`;
  document.getElementById("prodYearVal").innerHTML = `${year} <small>(${ageLabel})</small>`;
  document.getElementById("engineVolumeVal").textContent = `${Number(currentState.EngineVolume).toFixed(1)} L`;
  document.getElementById("cylindersVal").textContent = `${currentState.Cylinders} سلندر`;
  document.getElementById("mileageVal").textContent = `${Number(currentState.Mileage).toLocaleString("en-US")} km`;
  document.getElementById("levyVal").textContent = `$${Number(currentState.Levy).toLocaleString("en-US")}`;
  document.getElementById("airbagsVal").textContent = `${currentState.Airbags} وسائد`;

  document.getElementById("turboLabel").textContent = currentState.Turbo ? "مفعل (Turbo)" : "بدون تيربو";
  document.getElementById("leatherLabel").textContent = currentState.Leather ? "نعم" : "لا";
  document.getElementById("wheelLabel").textContent = currentState.Wheel ? "يمين" : "يسار";
}

function bindUIEvents() {
  // Progressive Disclosure Drawer ("مواصفات إضافية / Other")
  const advBtn = document.getElementById("toggleAdvancedBtn");
  const advPanel = document.getElementById("advancedPanel");
  advBtn.addEventListener("click", () => {
    const isExpanded = advBtn.getAttribute("aria-expanded") === "true";
    advBtn.setAttribute("aria-expanded", String(!isExpanded));
    advPanel.hidden = isExpanded;
  });

  const selectMap = {
    manufacturer: "Manufacturer",
    category: "Category",
    fuelType: "FuelType",
    gearBox: "GearBox"
  };
  Object.entries(selectMap).forEach(([id, key]) => {
    document.getElementById(id).addEventListener("change", (e) => {
      currentState[key] = e.target.value;
      clearActivePreset();
      updatePrediction();
    });
  });

  const sliderMap = {
    prodYear: "ProdYear",
    engineVolume: "EngineVolume",
    cylinders: "Cylinders",
    mileage: "Mileage",
    levy: "Levy",
    airbags: "Airbags"
  };
  Object.entries(sliderMap).forEach(([id, key]) => {
    document.getElementById(id).addEventListener("input", (e) => {
      currentState[key] = Number(e.target.value);
      clearActivePreset();
      syncSliderLabels();
      updatePrediction();
    });
  });

  const checkMap = {
    turbo: "Turbo",
    leather: "Leather",
    wheel: "Wheel"
  };
  Object.entries(checkMap).forEach(([id, key]) => {
    document.getElementById(id).addEventListener("change", (e) => {
      currentState[key] = e.target.checked;
      clearActivePreset();
      syncSliderLabels();
      updatePrediction();
    });
  });

  document.querySelectorAll(".segmented-control").forEach((group) => {
    const key = group.dataset.name;
    group.querySelectorAll(".seg-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        group.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        currentState[key] = key === "Doors" ? Number(btn.dataset.value) : btn.dataset.value;
        clearActivePreset();
        updatePrediction();
      });
    });
  });

  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const key = chip.dataset.preset;
      if (!PRESETS[key]) return;
      document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      currentState = { ...PRESETS[key] };
      applyStateToForm(currentState);
      updatePrediction();
    });
  });

  document.getElementById("resetBtn").addEventListener("click", () => {
    currentState = { ...PRESETS.toyota };
    document.querySelectorAll(".preset-chip").forEach((c) => {
      c.classList.toggle("active", c.dataset.preset === "toyota");
    });
    applyStateToForm(currentState);
    updatePrediction();
    showToast("تمت إعادة ضبط المواصفات");
  });

  document.getElementById("copyPriceBtn").addEventListener("click", () => {
    const priceText = document.getElementById("predictedPrice").textContent;
    const summary = `${currentState.Manufacturer} (${currentState.ProdYear}) - التقييم: $${priceText}`;
    navigator.clipboard?.writeText(summary);
    showToast("تم نسخ التقييم");
  });

  document.querySelectorAll(".nav-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      switchTab(tab.dataset.tab);
    });
  });

  document.getElementById("saveCompareBtn").addEventListener("click", () => {
    if (nextCompareSlot === "A") {
      compareSlotA = { ...currentState };
      nextCompareSlot = "B";
      showToast("تم حفظ السيارة في A");
    } else {
      compareSlotB = { ...currentState };
      nextCompareSlot = "A";
      showToast("تم حفظ السيارة في B");
    }
    renderComparison();
    switchTab("compare");
  });

  document.getElementById("loadCurrentToA").addEventListener("click", () => {
    compareSlotA = { ...currentState };
    renderComparison();
    showToast("تم تحديث السيارة A");
  });

  document.getElementById("loadCurrentToB").addEventListener("click", () => {
    compareSlotB = { ...currentState };
    renderComparison();
    showToast("تم تحديث السيارة B");
  });

  document.getElementById("themeToggle").addEventListener("click", () => {
    document.body.classList.toggle("light-glass");
  });
}

function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach((t) => {
    const active = t.dataset.tab === tabId;
    t.classList.toggle("active", active);
    t.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabId}`);
  });
}

function clearActivePreset() {
  document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
}

// ==========================================================================
// 5. Comparison Renderer
// ==========================================================================
function renderComparison() {
  if (!MODEL) return;
  const resA = predictCarPrice(compareSlotA);
  const resB = predictCarPrice(compareSlotB);

  document.getElementById("compareCardA").innerHTML = buildCompareColumnHTML("السيارة A", compareSlotA, resA);
  document.getElementById("compareCardB").innerHTML = buildCompareColumnHTML("السيارة B", compareSlotB, resB);

  const diff = resB.price - resA.price;
  const absDiff = Math.abs(Math.round(diff)).toLocaleString("en-US");
  const pct = ((Math.abs(diff) / Math.min(resA.price, resB.price)) * 100).toFixed(1);
  const higher = diff >= 0 ? "السيارة B أعلى سعراً بـ" : "السيارة A أعلى سعراً بـ";

  document.getElementById("compareSummaryCard").innerHTML = `
    <div class="vs-badge">VS</div>
    <span class="tag-pill">${higher}</span>
    <div class="diff-amount">$${absDiff}</div>
    <p style="color: var(--text-secondary); font-size: 0.88rem;">
      فارق بنسبة <strong>${pct}%</strong> وفق معاملات التقييم.
    </p>
  `;
}

function buildCompareColumnHTML(badge, car, res) {
  return `
    <span class="tag-pill">${badge}</span>
    <h3 style="margin-top: 12px;">${car.Manufacturer} (${car.ProdYear})</h3>
    <p style="color: var(--text-secondary); font-size: 0.84rem;">${car.Category} • ${car.FuelType}</p>
    <div class="compare-price">$${Math.round(res.price).toLocaleString("en-US")}</div>
    <ul class="compare-specs-list">
      <li><span>سعة المحرك</span><strong>${Number(car.EngineVolume).toFixed(1)}L ${car.Turbo ? "(Turbo)" : ""}</strong></li>
      <li><span>ناقل الحركة</span><strong>${car.GearBox}</strong></li>
      <li><span>نظام الدفع</span><strong>${car.DriveWheels}</strong></li>
      <li><span>المسافة المقطوعة</span><strong>${Number(car.Mileage).toLocaleString("en-US")} km</strong></li>
      <li><span>الأسطوانات / الوسائد</span><strong>${car.Cylinders} cyl / ${car.Airbags} airbags</strong></li>
      <li><span>الرسوم (Levy)</span><strong>$${car.Levy}</strong></li>
    </ul>
  `;
}

// ==========================================================================
// 6. Specular Rim Light Tracking
// ==========================================================================
function initSpecularGlassEffect() {
  const surfaces = document.querySelectorAll("[data-specular]");
  document.addEventListener("pointermove", (e) => {
    for (const el of surfaces) {
      const rect = el.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      el.style.setProperty("--mouse-x", `${x}px`);
      el.style.setProperty("--mouse-y", `${y}px`);
    }
  });
}

let toastTimer = null;
function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2400);
}
