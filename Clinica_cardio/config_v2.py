NIGHT_HOURS = [0, 1, 2, 3, 4, 5, 20, 22, 23]

# date format would be "YYYY-MM-DD"
# last baseline date must be the same
# date as start of study. Basically all
# dates must be mondays.
BASELINE = ['2023-01-01', '2023-03-31']
STUDY = ['2023-04-01', '2023-06-30']  # fechas generar informe
# PAST_WEEK = ['2023-01-30', '2023-02-06']

DATE_INTERVALS_TO_DISCARD = {
}

# variables that make up totalizer measurement
#ENERGY_VAR_LABELS = ('consumo-electricidad-dia')
#POWER_VAR_LABELS = ('consumo-agua-dia')


WHITELISTED_VAR_LABELS = (

    "consumo-electricidad-dia",
    "consumo-electricidad-mes",
    "energia-activa-hora-hora",
    "consumo-agua-dia",
    "consumo-agua-mes",
    "energia-termica-hora-hora",
    "energia-termica-mes",
    "t-out",
    "t-input",
    "cop_dia"

)


    #incluir temperaruta api-label : t-out
    #COP_DIA:  cop_dia