const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const score = document.getElementById("score");
const message = document.getElementById("message");
const error = document.getElementById("error");
const meterFill = document.getElementById("meterFill");
const predictBtn = document.getElementById("predictBtn");
const demoBtn = document.getElementById("demoBtn");
const againBtn = document.getElementById("againBtn");

const fields = ["study_hours","previous_score","attendance","sleep_hours","assignments_completed","practice_test_score"];

demoBtn.addEventListener("click", () => {
  const demo = {
    study_hours: 6, previous_score: 72, attendance: 85,
    sleep_hours: 7, assignments_completed: 8, practice_test_score: 75
  };
  fields.forEach(id => document.getElementById(id).value = demo[id]);
});

againBtn.addEventListener("click", () => {
  result.classList.add("hidden");
  document.getElementById("study_hours").focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";
  predictBtn.disabled = true;
  predictBtn.querySelector("span").textContent = "Running AI prediction...";

  const data = {};
  fields.forEach(id => data[id] = Number(document.getElementById(id).value));

  try {
   const response = await fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
    });

    const output = await response.json();
    if (!response.ok) throw new Error(output.detail || "Prediction failed.");

    const value = Number(output.predicted_score);
    score.textContent = value.toFixed(2);
    meterFill.style.width = `${value}%`;

    message.textContent =
      value >= 85 ? "Excellent predicted performance." :
      value >= 70 ? "Good predicted performance." :
      value >= 50 ? "Moderate predicted performance — more preparation may help." :
      "The prediction suggests the student may need additional preparation.";

    result.classList.remove("hidden");
    result.scrollIntoView({behavior:"smooth", block:"nearest"});
  } catch (err) {
    error.textContent = err.message || "Could not connect to the prediction service.";
  } finally {
    predictBtn.disabled = false;
    predictBtn.querySelector("span").textContent = "Predict final score";
  }
});
