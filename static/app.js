document.addEventListener("DOMContentLoaded", () => {
    // ------------------------------------------------------------------
    // Cache & Chart Registry
    // ------------------------------------------------------------------
    const API_BASE = ""; 
    let activeCharts = {};

    // Breadcrumbs Helper
    const breadcrumbCurrent = document.getElementById("active-breadcrumb");
    function updateBreadcrumb(name) {
        if (breadcrumbCurrent) {
            breadcrumbCurrent.textContent = name;
        }
    }

    // ------------------------------------------------------------------
    // Navigation / Routing Logic
    // ------------------------------------------------------------------
    const sections = document.querySelectorAll(".content-section");
    const navItems = document.querySelectorAll(".nav-item");

    function navigateToSection(sectionId, linkElement = null) {
        // Remove active classes
        navItems.forEach(nav => nav.classList.remove("active"));
        sections.forEach(sec => sec.classList.remove("active"));

        // Activate section
        const targetSec = document.getElementById(sectionId);
        if (targetSec) {
            targetSec.classList.add("active");
        }

        // Activate sidebar item
        if (linkElement) {
            linkElement.classList.add("active");
            updateBreadcrumb(linkElement.textContent.trim());
        } else {
            // Find sidebar link with matching data-section
            const matchingLink = Array.from(navItems).find(nav => nav.getAttribute("data-section") === sectionId);
            if (matchingLink) {
                matchingLink.classList.add("active");
                updateBreadcrumb(matchingLink.textContent.trim());
            }
        }
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    // Add navigation listeners
    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const targetSection = item.getAttribute("data-section");
            navigateToSection(targetSection, item);
        });
    });

    // Handle "Configure" dashboard shortcuts on Homepage
    document.addEventListener("click", (e) => {
        if (e.target && e.target.classList.contains("go-to-agent")) {
            const targetSec = e.target.getAttribute("data-target");
            navigateToSection(targetSec);
        }
    });

    // ------------------------------------------------------------------
    // API Helper with Error Boundary Handling
    // ------------------------------------------------------------------
    async function callAgentAPI(endpoint) {
        try {
            const response = await fetch(`${API_BASE}${endpoint}`);
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`API Error ${response.status}: ${errText || response.statusText}`);
            }
            return await response.json();
        } catch (e) {
            console.error("AI Agent Connection Failure:", e);
            return {
                status: "error",
                error_message: e.message || "Failed to reach AI Agent endpoint. Verify system network status."
            };
        }
    }

    // Load Dropdowns Options
    async function loadMetadata() {
        const vehicles = await callAgentAPI("/api/meta/vehicles");
        const vehicleSelect = document.getElementById("fleet-vehicle-select");
        if (vehicles && !vehicles.error_message && vehicleSelect) {
            vehicleSelect.innerHTML = vehicles.map(v => 
                `<option value="${v.vehicle_id}">${v.vehicle_id} (${v.current_vehicle_make} ${v.current_vehicle_model} - ${v.vehicle_type.replace('_', ' ')})</option>`
            ).join("");
        }

        const evs = await callAgentAPI("/api/meta/evs");
        const evSelect = document.getElementById("apm-ev-select");
        if (evs && !evs.error_message && evSelect) {
            evSelect.innerHTML = evs.map(id => `<option value="${id}">${id}</option>`).join("");
        }

        const batches = await callAgentAPI("/api/meta/batches");
        const batchSelect = document.getElementById("qms-batch-select");
        if (batches && !batches.error_message && batchSelect) {
            batchSelect.innerHTML = batches.map(b => `<option value="${b}">${b}</option>`).join("");
        }
    }

    loadMetadata();

    // ------------------------------------------------------------------
    // Visual Helpers (Loading, Error, Empty State)
    // ------------------------------------------------------------------
    function renderLoading(container, badge, statusMsg = "Analyzing operational parameters...") {
        if (badge) {
            badge.textContent = "Processing...";
            badge.className = "badge active";
        }
        container.innerHTML = `
            <div class="results-placeholder fade-in">
                <div class="placeholder-icon spinner"><i class="fa-solid fa-circle-notch fa-spin text-blue"></i></div>
                <h3>VoltGrid Agent Working</h3>
                <p class="loading-status-text">${statusMsg}</p>
                <div class="progress-bar-container">
                    <div class="progress-bar-fill"></div>
                </div>
            </div>
        `;
    }

    function renderError(container, badge, msg) {
        if (badge) {
            badge.textContent = "Failed";
            badge.className = "badge status-danger";
        }
        container.innerHTML = `
            <div class="error-container-card fade-in">
                <div class="error-header-row">
                    <i class="fa-solid fa-triangle-exclamation text-danger"></i>
                    <h4>AI Agent Execution Fault</h4>
                </div>
                <p class="error-detail-msg">${msg}</p>
                <div class="error-action-row">
                    <span>Possible Reason: Backend services or local datasets missing absolute path resolution. Check backend log stream.</span>
                </div>
            </div>
        `;
    }

    function destroyChart(key) {
        if (activeCharts[key]) {
            activeCharts[key].destroy();
            delete activeCharts[key];
        }
    }

    // ------------------------------------------------------------------
    // Agent 1: Fleet Electrification Readiness
    // ------------------------------------------------------------------
    const runFleetBtn = document.getElementById("run-fleet-agent-btn");
    const fleetContent = document.getElementById("fleet-results-content");
    const fleetBadge = document.getElementById("fleet-status-badge");

    if (runFleetBtn) {
        runFleetBtn.addEventListener("click", async () => {
            const vehicle = document.getElementById("fleet-vehicle-select").value;
            const query = document.getElementById("fleet-query-input").value;
            
            renderLoading(fleetContent, fleetBadge, "Evaluating EV powertrain match and ROI timelines...");

            const data = await callAgentAPI(`/api/agents/fleet_electrification?vehicle_id=${vehicle}&query=${encodeURIComponent(query)}`);
            
            if (data.status === "error" || !data.tool_outputs) {
                renderError(fleetContent, fleetBadge, data.error_message || "Could not retrieve electrification profile.");
                return;
            }

            if (fleetBadge) {
                fleetBadge.textContent = "Success";
                fleetBadge.className = "badge status-success";
            }

            const ready = data.tool_outputs.readiness_score_tool;
            const ev = data.tool_outputs.ev_matching_tool;
            const roi = data.tool_outputs.roi_tool;
            const proc = data.tool_outputs.procurement_tool;

            fleetContent.innerHTML = `
                <div class="ai-report fade-in">
                    <div class="report-summary-box">
                        <div class="report-summary-header">
                            <i class="fa-solid fa-robot"></i>
                            <strong>AI Agent System Summary</strong>
                        </div>
                        <p class="report-summary-text">${data.summary}</p>
                    </div>

                    <!-- Core metrics grid -->
                    <div class="report-grid">
                        <!-- Readiness Score Card -->
                        <div class="report-card flex-col align-center text-center">
                            <h4 class="report-card-title">Electrification Readiness</h4>
                            <p class="description-small">Composite score computed across vehicle operating habits.</p>
                            <div class="gauge-visual mt-3">
                                <div class="gauge-circle" style="--val: ${ready.readiness_score}">
                                    <div class="gauge-value">${ready.readiness_score}%</div>
                                </div>
                                <div class="gauge-label mt-2">${ready.classification}</div>
                            </div>
                        </div>

                        <!-- EV Match details -->
                        <div class="report-card">
                            <h4 class="report-card-title">EV Asset Replacement Match</h4>
                            <p class="description-small mb-3">Closest electrical vehicle replacement class match.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Recommended Model:</span>
                                <span class="metric-row-value text-blue">${ev.recommended_ev}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Battery Capacity:</span>
                                <span class="metric-row-value">${ev.battery_capacity_kwh} kWh</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Estimated Range:</span>
                                <span class="metric-row-value">${ev.estimated_range_km} km</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Compatibility Score:</span>
                                <span class="metric-row-value text-success">${(ev.compatibility_score * 100).toFixed(0)}%</span>
                            </div>
                            <p class="text-explanation mt-3">${ev.reason || ready.reason}</p>
                        </div>

                        <!-- Financial savings projections -->
                        <div class="report-card">
                            <h4 class="report-card-title">Financial Savings & ROI Analysis</h4>
                            <p class="description-small mb-3">Projected returns computed over active service life.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Total Annual Savings:</span>
                                <span class="metric-row-value text-success">$${roi.total_annual_savings_usd.toLocaleString()}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Annual Fuel Savings:</span>
                                <span class="metric-row-value">$${roi.annual_fuel_savings_usd.toLocaleString()}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Payback Period:</span>
                                <span class="metric-row-value">${roi.estimated_payback_years} Years</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">10-Year ROI:</span>
                                <span class="metric-row-value">${roi.roi_percent_over_10_years}%</span>
                            </div>
                        </div>
                    </div>

                    <!-- Visual Chart comparison and procurement analysis -->
                    <div class="report-grid">
                        <div class="report-card span-2">
                            <h4 class="report-card-title">Annual Operating Cost Comparison (USD)</h4>
                            <div class="chart-container" style="position: relative; height: 200px;">
                                <canvas id="fleet-cost-chart"></canvas>
                            </div>
                            <p class="description-small mt-2 text-center">Fuel Cost comparison shows a direct decrease in yearly energy overhead.</p>
                        </div>

                        <div class="report-card flex-col">
                            <h4 class="report-card-title">Procurement Timeline</h4>
                            <p class="description-small mb-3">Optimal buying window recommended by Carbon Agent.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Purchase Window:</span>
                                <span class="metric-row-value text-orange">${proc.recommended_purchase_window}</span>
                            </div>
                            <div class="metric-row flex-col mt-2">
                                <span class="metric-row-label">Action Priority Rating:</span>
                                <span class="metric-row-value text-blue" style="text-align: left; margin-top: 4px;">${proc.priority.toUpperCase()}</span>
                            </div>
                            <p class="text-explanation mt-3">${proc.reason}</p>
                        </div>
                    </div>

                    <!-- AI recommendations -->
                    <div class="report-grid">
                        <div class="report-card">
                            <h4 class="report-card-title"><i class="fa-solid fa-circle-check text-success"></i> Key AI Guidelines</h4>
                            <div class="ai-list">
                                ${data.recommendations.map(r => `
                                    <div class="ai-list-item rec">
                                        <i class="fa-solid fa-check"></i>
                                        <span>${r}</span>
                                    </div>
                                `).join("")}
                            </div>
                        </div>

                        <div class="report-card">
                            <h4 class="report-card-title"><i class="fa-solid fa-arrow-right-to-bracket text-blue"></i> Tactical Next Steps</h4>
                            <div class="ai-list">
                                ${data.next_steps.map(s => `
                                    <div class="ai-list-item step">
                                        <i class="fa-solid fa-circle-play"></i>
                                        <span>${s}</span>
                                    </div>
                                `).join("")}
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Draw comparison chart
            destroyChart('fleet-cost');
            const ctx = document.getElementById("fleet-cost-chart").getContext("2d");
            activeCharts['fleet-cost'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['ICE Vehicle (Fuel)', 'EV Match (Electricity)'],
                    datasets: [{
                        label: 'Annual Cost ($)',
                        data: [roi.estimated_annual_fuel_cost_usd, roi.estimated_annual_electricity_cost_usd],
                        backgroundColor: ['#f97316', '#10b981'],
                        borderWidth: 0,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        });
    }

    // ------------------------------------------------------------------
    // Agent 2: Maintenance Operations Optimiser
    // ------------------------------------------------------------------
    const runMaintBtn = document.getElementById("run-maint-agent-btn");
    const maintContent = document.getElementById("maint-results-content");
    const maintBadge = document.getElementById("maint-status-badge");

    if (runMaintBtn) {
        runMaintBtn.addEventListener("click", async () => {
            const query = document.getElementById("maint-query-input").value;
            renderLoading(maintContent, maintBadge, "Coordinating workshop spaces and vehicle priority...");

            const data = await callAgentAPI(`/api/agents/maintenance_operations?query=${encodeURIComponent(query)}`);

            if (data.status === "error" || !data.tool_outputs) {
                renderError(maintContent, maintBadge, data.error_message || "Could not coordinate maintenance schedule.");
                return;
            }

            if (maintBadge) {
                maintBadge.textContent = "Success";
                maintBadge.className = "badge status-success";
            }

            const risk = data.tool_outputs.maintenance_risk_analyzer;
            const schedule = data.tool_outputs.maintenance_schedule_optimizer;
            const planner = data.tool_outputs.charging_availability_planner;

            maintContent.innerHTML = `
                <div class="ai-report fade-in">
                    <div class="report-summary-box">
                        <div class="report-summary-header">
                            <i class="fa-solid fa-robot"></i>
                            <strong>AI Agent System Summary</strong>
                        </div>
                        <p class="report-summary-text">${data.summary}</p>
                    </div>

                    <div class="report-grid">
                        <!-- Risk Analyzer card -->
                        <div class="report-card">
                            <h4 class="report-card-title">Urgent Asset Risk Flag</h4>
                            <p class="description-small mb-3">Telemetry indicators calculated for active high-risk vehicle.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Vehicle ID:</span>
                                <span class="metric-row-value text-orange">${risk.vehicle_id}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Risk Severity Score:</span>
                                <span class="metric-row-value text-danger">${risk.risk_score}/100</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Risk Level:</span>
                                <span class="metric-row-value text-danger">${risk.risk_level}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Primary Factor:</span>
                                <span class="metric-row-value">${risk.dominant_risk_factor || 'Battery health degradation'}</span>
                            </div>
                            <p class="text-explanation mt-3">${risk.recommended_action}</p>
                        </div>

                        <!-- Workshop Booking Plan -->
                        <div class="report-card span-2">
                            <h4 class="report-card-title">Optimized Workshop Schedule Bookings</h4>
                            <p class="description-small mb-3">Vehicle repair queues sorted by priority and workshop workloads.</p>
                            <div class="table-container">
                                <table class="interactive-table">
                                    <thead>
                                        <tr>
                                            <th>Vehicle ID</th>
                                            <th>Assigned Workshop</th>
                                            <th>Day</th>
                                            <th>Time Slot</th>
                                            <th>Est. Downtime</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${schedule.map(s => `
                                            <tr>
                                                <td><strong>${s.vehicle_id}</strong></td>
                                                <td>${s.workshop_name}</td>
                                                <td>${s.scheduled_day}</td>
                                                <td class="text-blue">${s.scheduled_time_slot}</td>
                                                <td>${s.estimated_downtime_hours} hrs</td>
                                            </tr>
                                        `).join("")}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>

                    <!-- Charger availability & Post-maint scheduler -->
                    <div class="report-grid">
                        <div class="report-card">
                            <h4 class="report-card-title">Charger Availability Plan</h4>
                            <p class="description-small mb-3">Optimized charger reservation for vehicle post-service release.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Vehicle ID:</span>
                                <span class="metric-row-value">${planner.vehicle_id}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Target Charging Depot:</span>
                                <span class="metric-row-value">${planner.recommended_station}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Depot Location:</span>
                                <span class="metric-row-value">${planner.station_city}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Recommended Plug-In:</span>
                                <span class="metric-row-value text-success">${planner.charging_time}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Charger Class:</span>
                                <span class="metric-row-value">${planner.recommended_charger_class || 'Fast DC'}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Feasible in Window:</span>
                                <span class="metric-row-value">${planner.charging_feasible_in_window ? 'Yes' : 'No'}</span>
                            </div>
                        </div>

                        <!-- Data Chart for Priority level counts -->
                        <div class="report-card span-2">
                            <h4 class="report-card-title">Maintenance Priority Distribution</h4>
                            <div class="chart-container" style="position: relative; height: 200px;">
                                <canvas id="maint-priority-chart"></canvas>
                            </div>
                            <p class="description-small mt-2 text-center">Count of scheduled fleet assets grouped by risk priority level.</p>
                        </div>
                    </div>

                    <!-- Recommendations and Next Steps -->
                    <div class="report-grid">
                        <div class="report-card">
                            <h4 class="report-card-title"><i class="fa-solid fa-circle-check text-success"></i> Agent Recommendations</h4>
                            <div class="ai-list">
                                ${data.recommendations.map(r => `
                                    <div class="ai-list-item rec">
                                        <i class="fa-solid fa-check"></i>
                                        <span>${r}</span>
                                    </div>
                                `).join("")}
                            </div>
                        </div>
                        <div class="report-card">
                            <h4 class="report-card-title"><i class="fa-solid fa-arrow-right-to-bracket text-blue"></i> Execution Steps</h4>
                            <div class="ai-list">
                                ${data.next_steps.map(n => `
                                    <div class="ai-list-item step">
                                        <i class="fa-solid fa-circle-play"></i>
                                        <span>${n}</span>
                                    </div>
                                `).join("")}
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Draw Priority Chart
            destroyChart('maint-priority');
            const counts = { 'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0 };
            schedule.forEach(s => {
                const priority = s.priority || 'LOW';
                counts[priority] = (counts[priority] || 0) + 1;
            });

            const ctx = document.getElementById("maint-priority-chart").getContext("2d");
            activeCharts['maint-priority'] = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Object.keys(counts),
                    datasets: [{
                        label: 'Scheduled Vehicles',
                        data: Object.values(counts),
                        backgroundColor: ['#ef4444', '#f97316', '#3b82f6', '#10b981'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        });
    }

    // ------------------------------------------------------------------
    // Agent 3: Battery Health APM
    // ------------------------------------------------------------------
    const runApmBtn = document.getElementById("run-apm-agent-btn");
    const apmContent = document.getElementById("apm-results-content");
    const apmBadge = document.getElementById("apm-status-badge");

    if (runApmBtn) {
        runApmBtn.addEventListener("click", async () => {
            const evId = document.getElementById("apm-ev-select").value;
            renderLoading(apmContent, apmBadge, "Checking SOH degradation cycles and thermal telemetry logs...");

            const data = await callAgentAPI(`/api/agents/ev_apm?ev_id=${evId}`);

            if (data.status === "error" || !data.battery_analysis) {
                renderError(apmContent, apmBadge, data.error_message || "Could not retrieve battery telemetry.");
                return;
            }

            if (apmBadge) {
                apmBadge.textContent = "Success";
                apmBadge.className = "badge status-success";
            }

            apmContent.innerHTML = `
                <div class="ai-report fade-in">
                    <div class="report-summary-box">
                        <div class="report-summary-header">
                            <i class="fa-solid fa-robot"></i>
                            <strong>AI Agent Battery Summary</strong>
                        </div>
                        <p class="report-summary-text">${data.summary}</p>
                    </div>

                    <div class="report-grid">
                        <!-- SOH Gauge -->
                        <div class="report-card flex-col align-center text-center">
                            <h4 class="report-card-title">State of Health (SoH)</h4>
                            <p class="description-small">Total battery capacity retention relative to manufacturing specifications.</p>
                            <div class="gauge-visual mt-3">
                                <div class="gauge-circle" style="--val: ${data.battery_analysis.state_of_health_pct}">
                                    <div class="gauge-value">${data.battery_analysis.state_of_health_pct}%</div>
                                </div>
                                <div class="gauge-label mt-2">${data.battery_analysis.status}</div>
                            </div>
                        </div>

                        <!-- Thermal safety metrics -->
                        <div class="report-card">
                            <h4 class="report-card-title">Thermal Runaway Safety</h4>
                            <p class="description-small mb-3">Real-time operating temperature envelope analytics.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Average Operating Temp:</span>
                                <span class="metric-row-value">${data.safety_analysis.avg_temp_c}°C</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Max Recorded Temperature:</span>
                                <span class="metric-row-value">${data.safety_analysis.max_temp_c}°C</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Thermal Warnings:</span>
                                <span class="metric-row-value ${data.safety_analysis.thermal_runaway_warnings > 0 ? 'text-danger' : 'text-success'}">
                                    ${data.safety_analysis.thermal_runaway_warnings} anomalies
                                </span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Cooling Fan Status:</span>
                                <span class="metric-row-value">${data.safety_analysis.cooling_status}</span>
                            </div>
                        </div>

                        <!-- Telemetry usage metrics -->
                        <div class="report-card">
                            <h4 class="report-card-title">Cycle Usage Patterns</h4>
                            <p class="description-small mb-3">Daily battery charging habits and stress metrics.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Fast Charge Ratio:</span>
                                <span class="metric-row-value">${data.telemetry_data.fast_charge_ratio_pct}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Deep Discharge Cycles:</span>
                                <span class="metric-row-value">${data.telemetry_data.deep_discharge_cycles} cycles/mo</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Average Charge Duration:</span>
                                <span class="metric-row-value">${data.telemetry_data.avg_charge_duration_hours} hrs</span>
                            </div>
                        </div>
                    </div>

                    <!-- Line Chart showing degradation rate -->
                    <div class="report-grid">
                        <div class="report-card span-2">
                            <h4 class="report-card-title">Estimated SOH Degradation Curve (RUL Projection)</h4>
                            <div class="chart-container" style="position: relative; height: 220px;">
                                <canvas id="apm-degradation-chart"></canvas>
                            </div>
                            <p class="description-small mt-2 text-center">Remaining Useful Life (RUL) linear forecast. Critical replacement boundary at 70% SOH.</p>
                        </div>

                        <div class="report-card">
                            <h4 class="report-card-title"><i class="fa-solid fa-triangle-exclamation text-danger"></i> AI Predictive Warnings</h4>
                            <div class="ai-list mt-2">
                                ${data.maintenance_triggers.map(t => `
                                    <div class="ai-list-item rec">
                                        <i class="fa-solid fa-triangle-exclamation text-danger"></i>
                                        <span>${t}</span>
                                    </div>
                                `).join("")}
                            </div>
                        </div>
                    </div>

                    <div class="report-grid">
                        <div class="report-card span-3">
                            <h4 class="report-card-title"><i class="fa-solid fa-circle-check text-success"></i> Charge Cycle Guidelines</h4>
                            <div class="ai-list">
                                ${data.recommendations.map(r => `
                                    <div class="ai-list-item step">
                                        <i class="fa-solid fa-bolt text-blue"></i>
                                        <span>${r}</span>
                                    </div>
                                `).join("")}
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Draw Degradation Chart
            destroyChart('apm-degrad');
            const ctx = document.getElementById("apm-degradation-chart").getContext("2d");
            const months = [];
            const sohTrend = [];
            let currentSoh = data.battery_analysis.state_of_health_pct;
            const deg = data.battery_analysis.degradation_rate_monthly_pct;
            
            for (let i = 0; i <= 12; i++) {
                months.push(`Month ${i}`);
                sohTrend.push(Math.max(0, currentSoh - (deg * i)).toFixed(1));
            }

            activeCharts['apm-degrad'] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: months,
                    datasets: [
                        {
                            label: 'Projected SOH (%)',
                            data: sohTrend,
                            borderColor: '#8b5cf6',
                            borderWidth: 2,
                            tension: 0.1,
                            fill: false
                        },
                        {
                            label: 'Replacement Threshold (70%)',
                            data: Array(13).fill(70),
                            borderColor: '#ef4444',
                            borderDash: [5, 5],
                            borderWidth: 1.5,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#f8fafc' } } },
                    scales: {
                        y: { min: 60, max: 100, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        });
    }

    // ------------------------------------------------------------------
    // Agent 4: Cell Quality QMS
    // ------------------------------------------------------------------
    const runQmsBtn = document.getElementById("run-qms-agent-btn");
    const qmsContent = document.getElementById("qms-results-content");
    const qmsBadge = document.getElementById("qms-status-badge");

    if (runQmsBtn) {
        runQmsBtn.addEventListener("click", async () => {
            const batch = document.getElementById("qms-batch-select").value;
            renderLoading(qmsContent, qmsBadge, "Evaluating electrolyte filling anomalies and internal resistance drifts...");

            const data = await callAgentAPI(`/api/agents/ev_qms?batch_id=${batch}`);

            if (data.status === "error" || !data.cell_metrics) {
                renderError(qmsContent, qmsBadge, data.error_message || "Could not retrieve gigafactory QC analytics.");
                return;
            }

            if (qmsBadge) {
                qmsBadge.textContent = "Success";
                qmsBadge.className = "badge status-success";
            }

            qmsContent.innerHTML = `
                <div class="ai-report fade-in">
                    <div class="report-summary-box">
                        <div class="report-summary-header">
                            <i class="fa-solid fa-robot"></i>
                            <strong>AI Process Drift Report</strong>
                        </div>
                        <p class="report-summary-text">${data.quality_drift_analysis}</p>
                    </div>

                    <div class="report-grid">
                        <!-- Quality summary stats -->
                        <div class="report-card">
                            <h4 class="report-card-title">Production Summary</h4>
                            <p class="description-small mb-3">Inspected statistics computed over active production run.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Total Cells Inspected:</span>
                                <span class="metric-row-value">${data.cell_metrics.total_cells_inspected} cells</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Average Internal Resistance:</span>
                                <span class="metric-row-value text-blue">${data.cell_metrics.average_internal_resistance_mOhm} mΩ</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Average Cell Capacity:</span>
                                <span class="metric-row-value">${data.cell_metrics.average_cell_capacity_mAh} mAh</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Electrolyte Volume Average:</span>
                                <span class="metric-row-value">${data.cell_metrics.average_electrolyte_volume_ml} ml</span>
                            </div>
                        </div>

                        <!-- Scrap Rate gauge -->
                        <div class="report-card flex-col align-center text-center">
                            <h4 class="report-card-title">Production Scrap Rate</h4>
                            <p class="description-small">Total cells discarded relative to quality bounds.</p>
                            <div class="gauge-visual mt-3">
                                <div class="gauge-circle" style="--val: ${Math.min(100, data.cell_metrics.scrap_defect_rate_pct * 10)}">
                                    <div class="gauge-value">${data.cell_metrics.scrap_defect_rate_pct}%</div>
                                </div>
                                <div class="gauge-label mt-2">Control Boundary Limit < 2.0%</div>
                            </div>
                        </div>

                        <!-- Root Cause Audit -->
                        <div class="report-card">
                            <h4 class="report-card-title">AI Root Cause Diagnostics</h4>
                            <p class="description-small mb-3">Correlations computed across production parameters anomalies.</p>
                            <p class="text-explanation">${data.root_cause_analysis}</p>
                            <div class="ai-list mt-3">
                                ${data.alerts.map(a => `
                                    <div class="ai-list-item step">
                                        <i class="fa-solid fa-circle-exclamation text-orange"></i>
                                        <span>${a}</span>
                                    </div>
                                `).join("")}
                            </div>
                        </div>
                    </div>

                    <!-- Distributions charts -->
                    <div class="report-grid">
                        <div class="report-card span-2">
                            <h4 class="report-card-title">Grade Distribution (Cell Count)</h4>
                            <div class="chart-container" style="position: relative; height: 220px;">
                                <canvas id="qms-grades-chart"></canvas>
                            </div>
                        </div>

                        <div class="report-card">
                            <h4 class="report-card-title">Defect Type Distribution</h4>
                            <div class="chart-container" style="position: relative; height: 220px;">
                                <canvas id="qms-defects-chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Draw Grade chart
            destroyChart('qms-grades');
            const gCtx = document.getElementById("qms-grades-chart").getContext("2d");
            const gradeLabels = Object.keys(data.quality_distributions.grades);
            const gradeValues = Object.values(data.quality_distributions.grades);

            activeCharts['qms-grades'] = new Chart(gCtx, {
                type: 'bar',
                data: {
                    labels: gradeLabels,
                    datasets: [{
                        label: 'Cell Count',
                        data: gradeValues,
                        backgroundColor: ['#10b981', '#3b82f6', '#ef4444'],
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } }
                    }
                }
            });

            // Draw Defects chart
            destroyChart('qms-defects');
            const dCtx = document.getElementById("qms-defects-chart").getContext("2d");
            const defectLabels = Object.keys(data.quality_distributions.defect_categories);
            const defectValues = Object.values(data.quality_distributions.defect_categories);

            activeCharts['qms-defects'] = new Chart(dCtx, {
                type: 'doughnut',
                data: {
                    labels: defectLabels,
                    datasets: [{
                        data: defectValues,
                        backgroundColor: ['#ef4444', '#f97316', '#8b5cf6', '#3b82f6'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'right', labels: { color: '#f8fafc' } }
                    }
                }
            });
        });
    }

    // ------------------------------------------------------------------
    // Agent 5: Supply Chain Material Traceability
    // ------------------------------------------------------------------
    const runScBtn = document.getElementById("run-sc-agent-btn");
    const scContent = document.getElementById("sc-results-content");
    const scBadge = document.getElementById("sc-status-badge");

    if (runScBtn) {
        runScBtn.addEventListener("click", async () => {
            const query = document.getElementById("sc-query-input").value;
            renderLoading(scContent, scBadge, "Running trace on cobalt/lithium supply batches...");

            const data = await callAgentAPI(`/api/agents/supply_chain?query=${encodeURIComponent(query)}`);

            if (data.status === "error" || !data.supplier_details) {
                renderError(scContent, scBadge, data.error_message || "Could not retrieve supply chain logs.");
                return;
            }

            if (scBadge) {
                scBadge.textContent = "Success";
                scBadge.className = "badge status-success";
            }

            const details = data.supplier_details;
            const risk = data.mineral_risk;
            const quality = data.battery_quality;

            scContent.innerHTML = `
                <div class="ai-report fade-in">
                    <div class="report-summary-box">
                        <div class="report-summary-header">
                            <i class="fa-solid fa-robot"></i>
                            <strong>AI Audit Intelligence Report Summary</strong>
                        </div>
                        <p class="report-summary-text">${data.unified_report.replace(/\n/g, "<br>")}</p>
                    </div>

                    <!-- Node Flow Trace map -->
                    <div class="report-card">
                        <h4 class="report-card-title">Blockchain Material Traceability Node Map</h4>
                        <p class="description-small mb-4">Immutable chain logs verifying refinery and mining site origins.</p>
                        
                        <div class="timeline-flow">
                            ${data.traceability_nodes.map((n, i) => `
                                <div class="timeline-node ${i === 3 ? 'active' : 'success'}">
                                    <div class="timeline-icon">
                                        <i class="fa-solid ${i === 0 ? 'fa-mountain' : i === 1 ? 'fa-industry' : i === 2 ? 'fa-battery-full' : 'fa-truck-front'}"></i>
                                    </div>
                                    <span class="timeline-node-label">${n.label}</span>
                                    <span class="timeline-node-status">${n.status}</span>
                                </div>
                            `).join("")}
                        </div>
                    </div>

                    <div class="report-grid">
                        <!-- Supplier Profile -->
                        <div class="report-card">
                            <h4 class="report-card-title">Supplier Profile Audit</h4>
                            <p class="description-small mb-3">Basic vendor information from supplier database.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Refiner Name:</span>
                                <span class="metric-row-value text-blue">${details.supplier_name}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Refining Country:</span>
                                <span class="metric-row-value">${details.country}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Active Batch ID:</span>
                                <span class="metric-row-value text-orange">${details.batch_id}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Vendor Class Rating:</span>
                                <span class="metric-row-value">Tier ${details.supplier_tier} Vendor</span>
                            </div>
                        </div>

                        <!-- ESG Risk metrics -->
                        <div class="report-card">
                            <h4 class="report-card-title">ESG & Geopolitical Risk Rating</h4>
                            <p class="description-small mb-3">Geopolitical constraints calculated on mineral mining sites.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Global Mining Share:</span>
                                <span class="metric-row-value">${risk.global_supply_percentage}% Global</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Refiner Country:</span>
                                <span class="metric-row-value">${risk.country}</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Dependency Rating:</span>
                                <span class="metric-row-value">${risk.dependency_score}/100</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Geopolitical ESG Risk:</span>
                                <span class="metric-row-value text-danger">${risk.risk_level}</span>
                            </div>
                        </div>

                        <!-- Defect Rate logs -->
                        <div class="report-card">
                            <h4 class="report-card-title">Cell Quality Audit History</h4>
                            <p class="description-small mb-3">Gigafactory inspection stats mapped to raw materials vendor.</p>
                            <div class="metric-row">
                                <span class="metric-row-label">Total Inspected:</span>
                                <span class="metric-row-value">${quality.inspection_count} cells</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Material Defect Rate:</span>
                                <span class="metric-row-value text-danger">${(quality.defect_rate * 100).toFixed(1)}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-row-label">Common Defect:</span>
                                <span class="metric-row-value text-orange">${quality.defect_type}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
    }

    const scQuerySelect = document.getElementById("sc-query-select");
    const scQueryInput = document.getElementById("sc-query-input");
    if (scQuerySelect && scQueryInput) {
        scQuerySelect.addEventListener("change", () => {
            scQueryInput.value = scQuerySelect.value;
        });
    }

    // ------------------------------------------------------------------
    // Agent 6: Carbon Net Zero Tracker
    // ------------------------------------------------------------------
    const runCarbonBtn = document.getElementById("run-carbon-agent-btn");
    const carbonContent = document.getElementById("carbon-results-content");
    const carbonBadge = document.getElementById("carbon-status-badge");

    if (runCarbonBtn) {
        runCarbonBtn.addEventListener("click", async () => {
            const query = document.getElementById("carbon-query-input").value;
            renderLoading(carbonContent, carbonBadge, "Computing scope 1-3 footprint projections and emissions reductions...");

            const data = await callAgentAPI(`/api/agents/carbon_tracker?query=${encodeURIComponent(query)}`);

            if (data.status === "error" || !data.co2_history) {
                renderError(carbonContent, carbonBadge, data.error_message || "Could not compute carbon metrics.");
                return;
            }

            if (carbonBadge) {
                carbonBadge.textContent = "Success";
                carbonBadge.className = "badge status-success";
            }

            carbonContent.innerHTML = `
                <div class="ai-report fade-in">
                    <div class="report-summary-box">
                        <div class="report-summary-header">
                            <i class="fa-solid fa-robot"></i>
                            <strong>AI Greenhouse Gas Footprint Summary</strong>
                        </div>
                        <p class="report-summary-text">${data.unified_report}</p>
                    </div>

                    <div class="report-grid">
                        <!-- Line Chart Actual vs Target -->
                        <div class="report-card span-2">
                            <h4 class="report-card-title">Scope 1-3 GHG Emissions Trend (tCO2e)</h4>
                            <div class="chart-container" style="position: relative; height: 240px; width:100%">
                                <canvas id="carbon-trend-chart"></canvas>
                            </div>
                            <p class="description-small mt-2 text-center">Net Zero reduction path trajectory. Target baseline established in 2020.</p>
                        </div>

                        <!-- Target reductions status -->
                        <div class="report-card flex-col align-center text-center">
                            <h4 class="report-card-title">Verified Reductions</h4>
                            <p class="description-small">Total emissions reduction achieved relative to baseline year.</p>
                            <div class="gauge-visual mt-3">
                                <div class="gauge-circle" style="--val: ${data.carbon_reduction_summary_pct}">
                                    <div class="gauge-value">${data.carbon_reduction_summary_pct}%</div>
                                </div>
                                <div class="gauge-label mt-2">Overall GHG Reduced</div>
                            </div>
                        </div>
                    </div>

                    <!-- Scope Emission Table Factors -->
                    <div class="report-grid mt-4">
                        <div class="report-card span-3">
                            <h4 class="report-card-title">Scope 3 Raw Material Emission Factors (kg CO2e per kg)</h4>
                            <div class="table-container">
                                <table class="interactive-table">
                                    <thead>
                                        <tr>
                                            <th>Material Class</th>
                                            <th>Refiner / Vendor</th>
                                            <th>Scope 3 Factor</th>
                                            <th>Emissions Category</th>
                                            <th>Unit</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${data.emission_factors.map(f => `
                                            <tr>
                                                <td><strong>${f.material}</strong></td>
                                                <td>${f.supplier}</td>
                                                <td class="text-orange">${f.scope3_emission_factor_kg_co2_per_kg}</td>
                                                <td>${f.category}</td>
                                                <td>${f.unit}</td>
                                            </tr>
                                        `).join("")}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Draw line chart actual vs target
            destroyChart('carbon-trend');
            const ctx = document.getElementById("carbon-trend-chart").getContext("2d");
            const years = data.co2_history.map(h => h.year);
            const actuals = data.co2_history.map(h => h.total_emissions);
            const targets = data.co2_history.map(h => h.target_emissions);

            activeCharts['carbon-trend'] = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: years,
                    datasets: [
                        {
                            label: 'Actual Emissions',
                            data: actuals,
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.3
                        },
                        {
                            label: 'Net Zero Guideline',
                            data: targets,
                            borderColor: '#3b82f6',
                            borderDash: [5, 5],
                            borderWidth: 2,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { labels: { color: '#f8fafc' } }
                    },
                    scales: {
                        x: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        });
    }

    // ------------------------------------------------------------------
    // Central Supervisor Chatbot Integration
    // ------------------------------------------------------------------
    const chatBox = document.getElementById("supervisor-chat-box");
    const chatInput = document.getElementById("supervisor-chat-input");
    const chatSend = document.getElementById("supervisor-chat-send");
    const chatLoader = document.getElementById("supervisor-chat-loader");
    const chatLoaderMsg = document.getElementById("chat-loader-msg");

    function appendMessage(sender, text, isUser = false) {
        if (!chatBox) return;
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${isUser ? 'user' : 'assistant'} fade-in`;
        
        let formattedText = text;
        if (!isUser) {
            formattedText = formattedText
                .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/^- (.*$)/gim, '<li>$1</li>')
                .replace(/\n/g, '<br>');
        } else {
            formattedText = formattedText.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        }

        bubble.innerHTML = `
            <div class="chat-meta">
                <strong>${sender}</strong>
                <span>${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
            </div>
            <p>${formattedText}</p>
        `;
        chatBox.appendChild(bubble);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    async function handleSupervisorQuery(query) {
        if (!query.trim()) return;
        
        // Append user query bubble
        appendMessage("You", query, true);
        chatInput.value = "";
        
        // Determine parallel routing message
        const queryLower = query.toLowerCase();
        const targets = [];
        if (queryLower.match(/(readiness|transition|electrif|savings|roi|vh_|vehicle)/)) targets.push("Fleet Electrification");
        if (queryLower.match(/(maintenance|schedule|workload|workshop|downtime|risk)/)) targets.push("Maintenance Operations");
        if (queryLower.match(/(battery|apm|health|temperature|soh|degradation|ev-9)/)) targets.push("Battery Health APM");
        if (queryLower.match(/(qms|batch|cell|quality|drift|defect|electrolyte|bth-|batch-)/)) targets.push("Cell Quality QMS");
        if (queryLower.match(/(supply|trace|supplier|mineral|cobalt|nickel|lithium|sup-)/)) targets.push("Supply Chain");
        if (queryLower.match(/(carbon|emission|green|net-zero|scope)/)) targets.push("Carbon Net Zero");

        const targetDesc = targets.length > 0 ? targets.join(" & ") : "All 6 Agents";
        if (chatLoaderMsg) chatLoaderMsg.textContent = `Orchestrating [${targetDesc}] in parallel...`;
        if (chatLoader) chatLoader.classList.remove("hidden");

        // Call backend API
        const data = await callAgentAPI(`/api/agents/supervisor?query=${encodeURIComponent(query)}`);
        
        if (chatLoader) chatLoader.classList.add("hidden");

        if (data.status === "error" || !data.response) {
            appendMessage("AI Supervisor", `Sorry, I encountered an issue coordinating with the orchestration layers: ${data.error_message || 'Timeout'}`);
        } else {
            appendMessage("AI Supervisor", data.response);
        }
    }

    if (chatSend && chatInput) {
        chatSend.addEventListener("click", () => {
            handleSupervisorQuery(chatInput.value);
        });

        chatInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                handleSupervisorQuery(chatInput.value);
            }
        });
    }

    // Connect Presets click handlers
    document.querySelectorAll(".preset-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const query = btn.getAttribute("data-query");
            handleSupervisorQuery(query);
        });
    });
});
