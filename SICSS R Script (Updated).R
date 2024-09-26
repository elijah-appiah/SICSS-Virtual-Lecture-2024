##############################################################################
#                                                                            #
#                      BIG DATA MODELING WITH R                              #
#                                                                            #
##############################################################################

# Packages to Install and Load

install.packages("benchmarkme")   # benchmark for your computer
library(benchmarkme)

install.packages("tidyverse")  # contains packages for data analysis, visualization, etc.
library(tidyverse)

install.packages("data.table")  # big data manipulation with faster runtime
library(data.table)

install.packages("sparklyr")  # cluster computing platform for big data
library(sparklyr)

install.packages("ggExtra")  # supplement visualizations
library(ggExtra)

install.packages("GGally")  # for correlation matrix visualization
library(GGally)

install.packages("tidymodels")  # for machine learning modeling
library(tidymodels)

# One very important code to run to install Apache Spark locally in R
# Just run the code - no argument in the parenthesis
spark_install()


# Write code to generate sequence
# system.time()

# 1:10000000
# 
# seq(1, 10000000)
# 
# seq(from = 1, to = 10000000, by = 1)

generate_sequence_1 <- function(n){
  1:n
}


generate_sequence_2 <- function(n){
  seq(1, n)
}


generate_sequence_3 <- function(n){
  seq(from = 1, to = n, by = 1)
}

system.time(generate_sequence_1(10000000))
system.time(generate_sequence_2(10000000))
system.time(generate_sequence_3(10000000))


# Benchmarking - compare your system to other systems

?benchmark_std

results <- benchmark_std(runs = 3)
plot(results)
upload_results(results)


# data.table Package

?fread()

system.time({df <- read.csv("weatherAUS.csv")})  # `base` package
system.time({weather <- fread("weatherAUS.csv")})     # `data.table` package

str(weather)   # check structure of weather data
class(weather)  # what is the dataFrame type?

# DT[i, j, by]

unique(weather$Location)   # unique locations in Australia


albury <- weather[Location == "Albury"]  # filter rows for only Albury city
unique(albury$Location) 

cities_3 <- weather[Location %in% c("Albury", "Portland", "Adelaide")]
unique(cities_3$Location) 

max_temp_25 <- weather[MaxTemp > 25]

# Amount of Rainfall in Albury city
weather[Location == "Albury", "Rainfall"]  # returns a data.table (column emphasis)
weather[Location == "Albury", Rainfall]    # returns a vector

# Filter rows for Albury location, and have 3 columns
weather[Location == "Albury", c("Location", "Rainfall", "MaxTemp")]
weather[Location == "Albury", list(Location, Rainfall, MaxTemp)]
weather[Location == "Albury", .(Location, Rainfall, MaxTemp)]


weather[Location == "Albury", .(max_value = max(Rainfall, na.rm = TRUE))]

weather[Location == "Albury", .(max_value = max(Rainfall, na.rm = TRUE),
                                min_value = min(Rainfall, na.rm = TRUE),
                                average = mean(Rainfall, na.rm = TRUE))]


weather[Location == "Albury",
        .(average_rainfall = mean(Rainfall, na.rm = TRUE)), 
        by = RainToday]

weather[Location == "Albury",
        .(average_rainfall = mean(Rainfall, na.rm = TRUE)), 
        by = .(RainToday, RainTomorrow)]


weather[, .N, by = RainToday]

weather[, 
        hist(Rainfall)]


## APACHE SPARK ---- sparklyr Package in R

## Connect-work-disconnect

connection <- spark_connect(master = "local")

copy_to(dest = connection, df = weather)

src_tbls(connection)

weather %>% select(Location, Rainfall)

weather %>% filter(Location == "Albury")

weather %>% 
  filter(Location == "Albury", Temp3pm > 30) %>% 
  select(Location, Rainfall, RainToday) %>% 
  group_by(RainToday) %>% 
  summarise(average_rainfall = mean(Rainfall, na.rm = TRUE))


spark_disconnect(connection)

###################################################################

## Visualize the missing values in the dataset

install.packages("visdat")
library(visdat)

vis_miss(weather, warn_large_data = FALSE)


install.packages("dlookr")
library(dlookr)

plot_na_pareto(weather)


weather <- weather %>% 
  select(-c(Sunshine, 
            Evaporation, 
            Cloud3pm, 
            Cloud9am))


weather_df <- weather %>% drop_na()
plot_na_pareto(weather_df)
vis_miss(weather_df, warn_large_data = FALSE)



## VISUALIZATION with `ggplot2`

summary(weather_df$Rainfall)

ggplot(data = weather_df, aes(x = Rainfall)) +
  geom_histogram()

ggplot(data = weather_df %>% 
         filter(Rainfall <= 4,
                Location %in% c("Albury", "Portland", "Dartmoor")), 
       aes(x = Rainfall)) +
  geom_boxplot() +
  facet_wrap(~Location)


ggplot(weather_df %>% 
         filter(Location %in% c("Albury", 
                                "Portland", 
                                "Dartmoor",
                                "SydneyAirport",
                                "Darwin")), 
       aes(x = MaxTemp, fill = factor(Location))) +
  geom_histogram(color="black") +
  theme_bw()



#### MACHINE LEARNING

weather <- weather_df %>% 
  select(Rainfall, 
         MinTemp,
         MaxTemp,
         WindGustSpeed, 
         Humidity3pm, 
         Humidity9am,
         WindSpeed3pm,
         WindSpeed9am,
         Temp9am,
         Temp3pm,
         RainToday,
         RainTomorrow) %>% 
  mutate(RainToday = as.factor(RainToday),
         RainTomorrow = as.factor(RainTomorrow))

# Split data into Training and Testing Data --- rsample
weather_split <- initial_split(weather, prop = 0.80, strata = RainTomorrow)

weather_train <- training(weather_split)
weather_test <- testing(weather_split)


########### Model Specification   --- parsnip package
#1. Specify "model type"
#2. Set "engine"
#3. Set "mode" - regression or classification


# Baseline Logistic Regression Model

logit_model <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

# Fit the logistic regression model
logit_fit <- logit_model %>% 
  fit(RainTomorrow ~ ., data = weather)

tidy(logit_fit)
summary(logit_fit$fit)

# Make prediction

logit_pred <- logit_fit %>% 
  predict(new_data = weather_test)


weather_w_pred <- weather_test %>% 
  bind_cols(logit_pred)


weather_w_pred %>% 
  conf_mat(truth = RainTomorrow,
           estimate = .pred_class) %>% 
  autoplot(type = "heatmap")


weather_w_pred %>% 
  conf_mat(truth = RainTomorrow,
           estimate = .pred_class) %>% 
  autoplot(type = "mosaic")


weather_w_pred %>% 
  conf_mat(truth = RainTomorrow,
           estimate = .pred_class) %>% 
  summary()


## Another way of prediction

logit_pred_class <- logit_fit %>% 
  predict(new_data = weather_test,
          type = "class")

logit_pred_prob <- logit_fit %>% 
  predict(new_data = weather_test,
          type = "prob")

results_pred <- weather_test %>% 
  bind_cols(logit_pred_class, logit_pred_prob)


results_pred %>% 
  roc_curve(truth = RainTomorrow,
            .pred_Yes) %>% 
  autoplot()

results_pred %>% 
  roc_auc(truth = RainTomorrow,
          .pred_Yes)














