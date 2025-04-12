from application.api.services.disease_classification_service import classify_disease
from application.api.services.infection_prediction_service import predict_infection
from application.api.services.loss_prediction_service import predict_loss
from application.api.services.treatment_generation_service import generate_treatment
from application.api.services import user_configurations_service
import application.api.constants.diseases as diseases
from fastapi import HTTPException
import json

def generate_json_report(days, infected_areas, profit_loss, plantation_loss, yield_loss, classified_disease, disease_treatment):
    infection_by_day = []
    for i in range(len(days)):
        infection_by_day.append({
            "day": "{:.0f}".format(days[i]),
            "infected_area": "{:.0f}".format(infected_areas[i]) + " m²",
            "yield_loss": "{:.0f}".format(yield_loss[i]) + " tons",
            "economic_impact": {
                "profit_loss": "{:.2f}".format(profit_loss[i]) + " €",
                "plantation_loss": "{:.2f}".format(plantation_loss[i]) + " €"
            }
        })

    report = {
        "wheat_disease": classified_disease,
        "infection_by_day": infection_by_day,
        "disease_treatment": disease_treatment
    }
    return report

async def generate_report(image, user_id):

    configurations = await user_configurations_service.get_configuration_by_user_id(user_id)
    
    if not configurations.expected_yield or not configurations.expected_profit or not configurations.plantation_cost or not configurations.area:
        raise HTTPException(status_code=401, detail="Incomplete user configurations")
    
    classified_disease = classify_disease(image)
    if classified_disease == diseases.HEALTHY:
        return {
            "wheat_disease": diseases.HEALTHY
        }

    days, infected_areas = predict_infection(classified_disease, configurations.area)
    profit_loss, plantation_loss, yield_loss = predict_loss(
        configurations.area, infected_areas, configurations.plantation_cost, configurations.expected_profit, configurations.expected_yield
    )
    disease_treatment = generate_treatment(classified_disease)

    return generate_json_report(days, infected_areas, profit_loss,
                                 plantation_loss, yield_loss,
                                   classified_disease, disease_treatment)

# Example usage
if __name__ == "__main__":
    # Example image and area
    image = "path/to/image.jpg"
    area = 10000  # in square meters
    plantation_cost = 500000  # in euros
    expected_profit = 200000  # in euros
    expected_yield = 5000  # in tons

    report = generate_report(image, area, plantation_cost, expected_profit, expected_yield)
    print(json.dumps(report, indent=2))  # Pretty-print the report