# Grap the api key from the settings.json file
# remove everything before key as well as double quotes
COMET_API_KEY=$(grep 'apikey' settings.json | sed 's/^.*: //' | sed 's/"//g') \python -m comet_ml.scripts.comet_upload temp/offline_comet/*.zip
