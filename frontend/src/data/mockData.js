// Mock data extracted straight from the existing simulation's Streamlit state for perfect UI replication.

export const districtDemographics = [
    { district: "Thiruvananthapuram", population: 3301329, compliance: 0.82, econ: 0.85, icu: 450, cases: 330, vac: 1000 },
    { district: "Kollam", population: 2635375, compliance: 0.78, econ: 0.7, icu: 300, cases: 263, vac: 1000 },
    { district: "Pathanamthitta", population: 1197412, compliance: 0.85, econ: 0.65, icu: 150, cases: 119, vac: 1000 },
    { district: "Alappuzha", population: 2127789, compliance: 0.8, econ: 0.75, icu: 250, cases: 212, vac: 1000 },
    { district: "Kottayam", population: 1974551, compliance: 0.83, econ: 0.78, icu: 280, cases: 197, vac: 1000 },
    { district: "Idukki", population: 1108974, compliance: 0.75, econ: 0.6, icu: 100, cases: 110, vac: 1000 },
    { district: "Ernakulam", population: 3282388, compliance: 0.86, econ: 0.95, icu: 600, cases: 328, vac: 1000 },
    { district: "Thrissur", population: 3121200, compliance: 0.81, econ: 0.82, icu: 400, cases: 312, vac: 1000 },
    { district: "Palakkad", population: 2809934, compliance: 0.76, econ: 0.68, icu: 220, cases: 280, vac: 1000 },
    { district: "Malappuram", population: 4112920, compliance: 0.74, econ: 0.72, icu: 350, cases: 411, vac: 1000 },
    { district: "Kozhikode", population: 3086293, compliance: 0.79, econ: 0.8, icu: 450, cases: 308, vac: 1000 },
    { district: "Wayanad", population: 817420, compliance: 0.79, econ: 0.58, icu: 90, cases: 81, vac: 1000 },
    { district: "Kannur", population: 2523003, compliance: 0.8, econ: 0.74, icu: 300, cases: 252, vac: 1000 },
    { district: "Kasaragod", population: 1307375, compliance: 0.77, econ: 0.62, icu: 140, cases: 130, vac: 1000 }
];

// Recreating the exact outbreak curve numbers (approx) over 100 days arriving at final 738,825
export const outbreakCurveData = Array.from({length: 100}, (_, i) => {
    // Exponential curve y = a * e^(bx) ending approx at 738,825
    const val = 3000 * Math.exp(0.055 * i);
    return { day: i, cases: Math.floor(val) };
});

export const finalDirectives = [
    { district: "Thiruvananthapuram", policy: "Level 1", cases: 75822, pop: 3301329, lat: 8.5241, lon: 76.9366 },
    { district: "Kollam", policy: "Level 3", cases: 50562, pop: 2635375, lat: 8.8932, lon: 76.6141 },
    { district: "Pathanamthitta", policy: "Level 2", cases: 16858, pop: 1197412, lat: 9.2648, lon: 76.7870 },
    { district: "Alappuzha", policy: "Level 2", cases: 68815, pop: 2127789, lat: 9.4981, lon: 76.3388 },
    { district: "Kottayam", policy: "Level 3", cases: 18858, pop: 1974551, lat: 9.5916, lon: 76.5222 },
    { district: "Idukki", policy: "Level 0", cases: 37315, pop: 1108974, lat: 9.8500, lon: 76.9492 },
    { district: "Ernakulam", policy: "Level 3", cases: 42086, pop: 3282388, lat: 9.9816, lon: 76.2999 },
    { district: "Thrissur", policy: "Level 1", cases: 69832, pop: 3121200, lat: 10.5276, lon: 76.2144 },
    { district: "Palakkad", policy: "Level 3", cases: 60884, pop: 2809934, lat: 10.7867, lon: 76.6548 },
    { district: "Malappuram", policy: "Level 2", cases: 60466, pop: 4112920, lat: 11.0733, lon: 76.0740 },
    { district: "Kozhikode", policy: "Level 3", cases: 88412, pop: 3086293, lat: 11.2588, lon: 75.7804 },
    { district: "Wayanad", policy: "Level 2", cases: 34110, pop: 817420, lat: 11.6854, lon: 76.1320 },
    { district: "Kannur", policy: "Level 3", cases: 72892, pop: 2523003, lat: 11.8745, lon: 75.3704 },
    { district: "Kasaragod", policy: "Level 2", cases: 42913, pop: 1307375, lat: 12.4962, lon: 74.9869 }
];

export const logisticsData = [
    { district: "Thiruvananthapuram", cap: 146.3, used: 658, limit: 450, o2: -416 },
    { district: "Kollam", cap: 150.0, used: 450, limit: 300, o2: -300 },
    { district: "Pathanamthitta", cap: 118.4, used: 177, limit: 150, o2: -55 },
    { district: "Alappuzha", cap: 146.5, used: 366, limit: 250, o2: -232 },
    { district: "Kottayam", cap: 131.9, used: 369, limit: 280, o2: -178 },
    { district: "Idukki", cap: 135.5, used: 135, limit: 100, o2: -70 },
    { district: "Ernakulam", cap: 118.6, used: 711, limit: 600, o2: -223 },
    { district: "Thrissur", cap: 150.0, used: 600, limit: 400, o2: -400 },
    { district: "Palakkad", cap: 100.6, used: 221, limit: 220, o2: -2 },
    { district: "Malappuram", cap: 131.3, used: 459, limit: 350, o2: -218 },
    { district: "Kozhikode", cap: 113.9, used: 512, limit: 450, o2: -125 },
    { district: "Wayanad", cap: 135.0, used: 121, limit: 90, o2: -63 },
    { district: "Kannur", cap: 123.7, used: 371, limit: 300, o2: -142 },
    { district: "Kasaragod", cap: 138.0, used: 193, limit: 140, o2: -106 }
];
