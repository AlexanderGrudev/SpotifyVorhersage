library(caret)
library(randomForest)
library(xgboost)
library(dplyr)
library(ggplot2)
library(readr)
library(corrplot)
library(fastDummies)
library(modeldata)
library(doParallel)
library(tidymodels)
dataset <- read_csv("Transfermodul/dataset.csv")

#Überblick Datenset
head(dataset)
str(dataset)
summary(dataset)

 
# Histogramm der Popularität
ggplot(dataset, aes(x = popularity)) +
  geom_histogram(binwidth = 5, fill = 'blue', color = 'black', alpha = 0.7) +
  ggtitle('Verteilung der Popularität') +
  xlab('Popularität') +
  ylab('Häufigkeit') +
  theme_minimal()

# Korrelation 
corr_matrix <- cor(dataset %>% select(popularity,duration_ms,danceability,energy,energy,
                                      loudness,speechiness,acousticness,tempo), use = "complete.obs")
corrplot(corr_matrix, method = "circle")





# Duplikate erkennen
dataset %>%
  group_by(track_name, artists) %>%  
  summarise(count = n()) %>%  
  arrange(desc(count)) %>%  
  head(20)  

#Duplikate entfernen 

data <- dataset %>%
  group_by(track_name, artists) %>%
  slice(1) %>%  
  ungroup()  

# Überprüfen 
data %>%
  group_by(track_name, artists) %>%  
  summarise(count = n()) %>%  
  arrange(desc(count)) %>%  
  head(20)  

# Popularität = 0 entfernen
data <- data[data$popularity != 0, ]

# Umwandeln von benötigten Textdaten in Nummerische Werte 
# Anzahl der Künstler
data$artist_count <- sapply(strsplit(as.character(data$artists), ";"), length)

# Länge des Songtitels und Albumnamens
data$title_length <- nchar(data$track_name)
data$album_length <- nchar(data$album_name)

# Den ersten Künstler extrahieren
data$artists <- sapply(strsplit(data$artists, ";"), `[`, 1)

# Durchschnittliche Popularität pro Künstler berechnen
artist_stats <- data %>%
  group_by(artists) %>%
  summarise(mean_popularity_artist = mean(popularity, na.rm = TRUE))

data <- left_join(data, artist_stats, by = "artists")

# Explicit als Dummy

data <- data %>% 
  mutate(
    explicit = ifelse(explicit == "TRUE", 1, 0))


# Entfernen der restlichen nicht nummerischen Spalten 
data <- data[, sapply(data, is.numeric)]
data <- data %>% 
  select(-...1)

# Bei Nullwerten den Durchschnitt nehmen 

# Berechnung des Durchschnitts (Mittelwert) ohne die 0-Werte
mean_dance <- mean(data$danceability[data$danceability != 0], na.rm = TRUE)
mean_valence <- mean(data$valence[data$valence != 0], na.rm = TRUE)
mean_tempo <- mean(data$tempo[data$tempo != 0], na.rm = TRUE)
mean_energy <- mean(data$energy[data$energy != 0], na.rm = TRUE)

# Ersetzen der 0-Werte durch den berechneten Durchschnitt
data$danceability[data$danceability == 0] <- mean_dance
data$valence[data$valence == 0] <- mean_valence
data$tempo[data$tempo == 0] <- mean_tempo
data$energy[data$energy == 0] <- mean_energy

# Werte runden um Kapazität nicht auszulasten 

data <- data %>%
  mutate(
    danceability = round(danceability, 3),
    energy = round(energy, 3),
    loudness = round(loudness, 3),
    speechiness = round(speechiness, 3),
    acousticness = round(acousticness, 3),
    instrumentalness = round(instrumentalness, 3),
    liveness = round(liveness, 3),
    valence = round(valence, 3),
    tempo = round(tempo, 2),
    duration_ms = duration_ms/60000,
    duration_ms = round(duration_ms, 2),
   mean_popularity_artist = round(mean_popularity_artist, 1)
  )

 

# NA Werte erkennen und entfernen

na_count <- colSums(is.na(data))
print(na_count)

data <- data %>% 
  mutate(across(everything(), ~ replace_na(.x, 0)))

na_count <- colSums(is.na(data))
print(na_count)


# Daten als numeric speichern für weitere Berechnungen
data<- data %>% mutate_all(as.numeric)


str(data)

# Daten aufspliten
set.seed(12345)
data_split <- initial_split(data, prop = 0.8)
train_data <- training(data_split)
test_data <- testing(data_split)

# Verkürzung wegen Serverkapazität
set.seed(12345)
pretrain_data <- train_data[sample(nrow(train_data), 10000), ]
pretrain_data <- na.omit(pretrain_data)

pretest_data <- test_data[sample(nrow(test_data), 2000),]
pretest_data <- na.omit(pretest_data)

# a. Lineare Regression
lm_model <- lm(popularity ~ ., data = pretrain_data)

# b. Random Forest
rf_model <- randomForest(popularity ~ ., data = pretrain_data, 
                         ntree = 250, importance = TRUE)

# 5. Modellbewertung

# a. Lineare Regression Vorhersagen und Bewertung
lm_predictions <- predict(lm_model, newdata = pretest_data)

# b. Random Forest Vorhersagen und Bewertung
rf_predictions <- predict(rf_model, newdata = pretest_data)


# Funktion zur Berechnung der Metriken
evaluate_model <- function(actual, predicted, model_name) {
  mae <- MAE(predicted, actual)
  rmse <- RMSE(predicted, actual)
  r2 <- R2(predicted, actual)
  cat(paste("Model:", model_name, "\n"))
  cat(paste("MAE:", round(mae, 2), "\n"))
  cat(paste("RMSE:", round(rmse, 2), "\n"))
  cat(paste("R-squared:", round(r2, 2), "\n\n"))
}

# Bewertung der Modelle
evaluate_model(pretest_data$popularity, lm_predictions, "Lineare Regression")
evaluate_model(pretest_data$popularity, rf_predictions, "Random Forest")





cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Random Forest Hyperparameter-Tuning

rf_tuned <- train(
  popularity ~ .,
  data = pretrain_data,
  method = "rf",
  ntree = 500,  
  trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3),  
  tuneGrid = expand.grid(mtry = c(2, 4, 6, 8))  
)


# Bewertung des getunten Random Forest
rf_tuned_predictions <- predict(rf_tuned, newdata = pretest_data)
evaluate_model(pretest_data$popularity, rf_tuned_predictions, "Random Forest (Getunt)")

# XGBoost 
# XGBosst-Rezept erstellen
data_recipe <- recipe(popularity ~ ., data = pretrain_data)

# XGBoost-Modell definieren
xgb_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  loss_reduction = tune(), 
  min_n = tune(),  
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Grid definieren
# Erweitertes Grid für XGBoost 
xgb_grid <- grid_regular(
  trees(range = c(50, 200)),                
  learn_rate(range = c(0.05, 0.2)),         
  tree_depth(range = c(3, 6)),              
  loss_reduction(range = c(0, 5)),          
  min_n(range = c(1, 5)),                   
  levels = 3                                 
)

# Cross-Validation definieren
cv_folds <- vfold_cv(pretrain_data, v = 3)

# Workflow erstellen
xgb_wf <- workflow() %>%
  add_recipe(data_recipe) %>% 
  add_model(xgb_model)
  

# Tuning
xgb_res <- tune_grid(
  xgb_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)


stopCluster(cl)

# Ergebnisse anzeigen
collect_metrics(xgb_res)

# Beste Parameter auswählen
best_params <- xgb_res %>% 
  select_best(metric = "rsq")  


# Finales Modell erstellen
final_wf <- finalize_workflow(
  xgb_wf,
  best_params
)

# Modell auf den vollständigen Trainingsdaten anpassen
final_fit <- fit(final_wf, data = pretrain_data)

# Evaluierung auf Testdaten 
test_results <- predict(final_fit, new_data = pretest_data) %>%
  bind_cols(pretest_data) %>%
  metrics(truth = popularity, estimate = .pred)

print(test_results)



# Koeffizienten extrahieren und nach Wichtigkeit ordnen
lm_importance <- summary(lm_model)$coefficients
lm_importance <- as.data.frame(lm_importance)
lm_importance <- lm_importance[order(abs(lm_importance[, "Estimate"]), decreasing = TRUE), ]


print(lm_importance)



