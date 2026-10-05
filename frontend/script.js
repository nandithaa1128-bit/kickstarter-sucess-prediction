// ============================================================
// API URL
// ============================================================

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// ELEMENTS
// ============================================================

const form = document.getElementById("predictionForm");

const categorySelect =
    document.getElementById("category");

const mainCategorySelect =
    document.getElementById("main_category");

const currencySelect =
    document.getElementById("currency");

const countrySelect =
    document.getElementById("country");

const resultSection =
    document.getElementById("resultSection");

const errorMessage =
    document.getElementById("errorMessage");

const predictButton =
    document.getElementById("predictButton");

const buttonText =
    document.getElementById("buttonText");

const buttonLoader =
    document.getElementById("buttonLoader");


// ============================================================
// ADD OPTIONS TO SELECT
// ============================================================

function populateSelect(selectElement, values, placeholder) {

    selectElement.innerHTML = "";

    const defaultOption =
        document.createElement("option");

    defaultOption.value = "";

    defaultOption.textContent = placeholder;

    selectElement.appendChild(defaultOption);


    values.forEach(value => {

        const option =
            document.createElement("option");

        option.value = value;

        option.textContent = value;

        selectElement.appendChild(option);

    });
}


// ============================================================
// LOAD MODEL OPTIONS
// ============================================================

async function loadOptions() {

    try {

        const response =
            await fetch(`${API_URL}/options`);

        if (!response.ok) {

            throw new Error(
                "Unable to load model options."
            );

        }

        const data =
            await response.json();


        populateSelect(
            categorySelect,
            data.categories,
            "Select category"
        );


        populateSelect(
            mainCategorySelect,
            data.main_categories,
            "Select main category"
        );


        populateSelect(
            currencySelect,
            data.currencies,
            "Select currency"
        );


        populateSelect(
            countrySelect,
            data.countries,
            "Select country"
        );

    }

    catch (error) {

        showError(
            "Could not connect to the prediction server. " +
            "Make sure FastAPI is running."
        );

        console.error(error);
    }
}


// ============================================================
// ERROR
// ============================================================

function showError(message) {

    errorMessage.textContent = message;

    errorMessage.style.display = "block";
}


function hideError() {

    errorMessage.style.display = "none";
}


// ============================================================
// LOADING STATE
// ============================================================

function setLoading(isLoading) {

    predictButton.disabled = isLoading;

    if (isLoading) {

        buttonText.textContent =
            "Analyzing Project...";

        buttonLoader.classList.remove(
            "hidden"
        );

    }

    else {

        buttonText.textContent =
            "Predict Success";

        buttonLoader.classList.add(
            "hidden"
        );
    }
}


// ============================================================
// SHOW RESULT
// ============================================================

function showResult(data) {

    const isSuccessful =
        data.prediction === 1;


    const statusClass =
        isSuccessful
            ? "success"
            : "failure";


    const statusIcon =
        isSuccessful
            ? "✓"
            : "✕";


    resultSection.innerHTML = `

        <div class="result-content">

            <div class="result-label">
                PREDICTED SUCCESS PROBABILITY
            </div>


            <div class="probability">
                ${data.probability_percent.toFixed(2)}%
            </div>


            <div class="result-status ${statusClass}">
                ${statusIcon}
                ${data.result}
            </div>


            <div class="probability-track">

                <div
                    class="probability-fill"
                    style="width: 0%"
                    data-width="${data.probability_percent}%"
                ></div>

            </div>


            <div class="probability-caption">

                <span>
                    0%
                </span>

                <span>
                    Threshold: ${data.threshold_percent}%
                </span>

                <span>
                    100%
                </span>

            </div>


            <div class="result-details">

                <div class="detail-box">

                    <span>
                        Duration
                    </span>

                    <strong>
                        ${data.duration_days} days
                    </strong>

                </div>


                <div class="detail-box">

                    <span>
                        Category
                    </span>

                    <strong>
                        ${data.category}
                    </strong>

                </div>


                <div class="detail-box">

                    <span>
                        Country
                    </span>

                    <strong>
                        ${data.country}
                    </strong>

                </div>


                <div class="detail-box">

                    <span>
                        Threshold
                    </span>

                    <strong>
                        ${data.threshold_percent}%
                    </strong>

                </div>

            </div>


            <button
                class="try-again"
                onclick="resetPrediction()"
            >
                ← Predict Another Project
            </button>

        </div>
    `;


    // Animate probability bar

    setTimeout(() => {

        const fill =
            resultSection.querySelector(
                ".probability-fill"
            );

        if (fill) {

            fill.style.width =
                fill.dataset.width;
        }

    }, 100);
}


// ============================================================
// FORM SUBMISSION
// ============================================================

form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();

        hideError();

        setLoading(true);


        const projectData = {

            name:
                document.getElementById(
                    "name"
                ).value.trim(),

            goal:
                Number(
                    document.getElementById(
                        "goal"
                    ).value
                ),

            usd_goal_real:
                Number(
                    document.getElementById(
                        "usd_goal_real"
                    ).value
                ),

            launch_date:
                document.getElementById(
                    "launch_date"
                ).value,

            deadline_date:
                document.getElementById(
                    "deadline_date"
                ).value,

            category:
                categorySelect.value,

            main_category:
                mainCategorySelect.value,

            currency:
                currencySelect.value,

            country:
                countrySelect.value
        };


        try {

            const response =
                await fetch(
                    `${API_URL}/predict`,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                projectData
                            )
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Prediction failed."
                );

            }


            showResult(data);

        }

        catch (error) {

            showError(
                error.message
            );

            console.error(error);

        }

        finally {

            setLoading(false);

        }

    }
);


// ============================================================
// RESET
// ============================================================

function resetPrediction() {

    resultSection.innerHTML = `

        <div class="result-placeholder">

            <div class="result-icon">
                ✦
            </div>

            <h3>
                Your prediction
                <br>
                will appear here
            </h3>

            <p>
                Complete the project details
                and click Predict Success.
            </p>

        </div>
    `;
}


// ============================================================
// START
// ============================================================

loadOptions();