########################################################
###### TRABALHO DE CONCLUSÃO DE CURSO ##################
###### ALEXANDRE AGUIAR GUERREIRO ######################
############### USP ####################################
########################################################

# Carregar bibliotecas necessárias
library(randomForest)
library(xgboost)
library(dplyr)
library(lubridate)
library(caret)
library(Matrix)
library(readxl)
library(ggplot2)
library(car)
library(ggcorrplot)
library(tidyr)
library(kableExtra)
library(randomForest)

# Carregar os dados
assistencias4 <- read_excel("assistencias-EJA-final.xlsx")

# Manipular variáveis de data e hora
assistencias4 <- assistencias4 %>%
  mutate(DATA_YEAR = year(DATA),
         DATA_MONTH = month(DATA),
         DATA_DAY = day(DATA),
         HORAJOGO_HOUR = hour(HORAJOGO),
         HORAJOGO_MINUTE = minute(HORAJOGO)) %>%
  select(-DATA, -HORAJOGO)

# Codificar variáveis categóricas
assistencias4$EPOCA <- as.factor(assistencias4$EPOCA)
assistencias4$VISITANTE <- as.factor(assistencias4$VISITANTE)
assistencias4$CHUVAJOGO <- as.factor(assistencias4$CHUVAJOGO)
assistencias4$CHUVA1HJOGO <- as.factor(assistencias4$CHUVA1HJOGO)
assistencias4$CHUVA2HJOGO <- as.factor(assistencias4$CHUVA2HJOGO)
assistencias4$COBERTURAJOGO <- as.factor(assistencias4$COBERTURAJOGO)
assistencias4$COBERTURA1HJOGO <- as.factor(assistencias4$COBERTURA1HJOGO)
assistencias4$COBERTURA2HJOGO <- as.factor(assistencias4$COBERTURA2HJOGO)
assistencias4$TV <- as.factor(assistencias4$TV)

# Limitar a variável dependente ASSISTENCIA a 50045
assistencias4$ASSISTENCIA <- pmin(assistencias4$ASSISTENCIA, 50045)

# Imputar valores ausentes se necessário
assistencias4[is.na(assistencias4)] <- 0  # Exemplo simples de imputação

# Remover variáveis recomendadas
assistencias4 <- assistencias4 %>%
  select(-PONTOS, -TEMP1HJOGO, -PONTOSVISIT, -PONTOSLIDER, -DATA_MONTH)  # Remover as variáveis recomendadas

# Análise de Correlação
correlation_matrix <- cor(assistencias4 %>% select_if(is.numeric))
print("Matriz de correlação:")
print(correlation_matrix)

# Plotar a Matriz de Correlação
ggcorrplot(correlation_matrix, 
           hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           title = "Matriz de Correlação", 
           ggtheme = ggplot2::theme_minimal())

# Análise de Condicionamento (Condition Number)
X_matrix <- model.matrix(~ ., data = assistencias4 %>% select(-ASSISTENCIA))
svd_decomp <- svd(X_matrix)
condition_number <- max(svd_decomp$d) / min(svd_decomp$d)

cat("Número de Condição (Condition Number):", condition_number, "\n")
if (condition_number > 30) {
  cat("Aviso: Alto número de condição indica possível multicolinearidade!\n")
}

# Definir controle para validação cruzada (5-fold CV)
train_control <- trainControl(method = "cv", number = 5)

# Função para treinar modelo Random Forest com hiperparâmetros ajustados
train_rf <- function(data) {
  rf_model <- randomForest(ASSISTENCIA ~ ., data = data, ntree = 200, mtry = 3)
  return(rf_model)
}

# Função para treinar modelo XGBoost com ajustes de hiperparâmetros
train_xgb <- function(data) {
  X_matrix <- model.matrix(~ . + 0, data = data %>% select(-ASSISTENCIA))
  y <- data$ASSISTENCIA
  dtrain <- xgb.DMatrix(data = X_matrix, label = y)
  
  params <- list(
    booster = "gbtree", 
    objective = "reg:squarederror",
    eta = 0.05,  
    max_depth = 7,  
    subsample = 0.9,  
    colsample_bytree = 0.7  
  )
  
  xgb_model <- xgb.train(
    params = params, 
    data = dtrain, 
    nrounds = 150,  
    watchlist = list(train = dtrain),
    verbose = 0
  )
  return(xgb_model)
}

# Função para treinar modelo de Regressão Linear
train_lm <- function(data) {
  lm_model <- lm(ASSISTENCIA ~ ., data = data)
  return(lm_model)
}

# Treinar múltiplos modelos Random Forest
rf_models <- lapply(1:5, function(i) {
  train_rf(assistencias4)
})

# Treinar múltiplos modelos XGBoost com os ajustes
xgb_models <- lapply(1:5, function(i) {
  train_xgb(assistencias4)
})

# Treinar modelo de Regressão Linear
lm_model <- train_lm(assistencias4)

# Fazer previsões com todos os modelos
rf_predictions <- lapply(rf_models, function(model) {
  predict(model, newdata = assistencias4)
})

xgb_predictions <- lapply(xgb_models, function(model) {
  X_test_matrix <- model.matrix(~ . + 0, data = assistencias4 %>% select(-ASSISTENCIA))
  dtest <- xgb.DMatrix(data = X_test_matrix)
  
  # Fazer a previsão
  predict(model, dtest)
})

lm_predictions <- predict(lm_model, newdata = assistencias4)

# Limitar as previsões a 50045
rf_avg_predictions <- pmin(rowMeans(do.call(cbind, rf_predictions)), 50045)
xgb_avg_predictions <- pmin(rowMeans(do.call(cbind, xgb_predictions)), 50045)
lm_predictions <- pmin(lm_predictions, 50045)

# Calcular RMSE e R-squared para Random Forest
rf_rmse <- sqrt(mean((rf_avg_predictions - assistencias4$ASSISTENCIA)^2))
rf_r2 <- 1 - (sum((assistencias4$ASSISTENCIA - rf_avg_predictions)^2) / sum((assistencias4$ASSISTENCIA - mean(assistencias4$ASSISTENCIA))^2))

print(paste("Random Forest RMSE:", rf_rmse))
print(paste("Random Forest R-squared:", rf_r2))

# Calcular RMSE e R-squared para XGBoost
xgb_rmse <- sqrt(mean((xgb_avg_predictions - assistencias4$ASSISTENCIA)^2))
xgb_r2 <- 1 - (sum((assistencias4$ASSISTENCIA - xgb_avg_predictions)^2) / sum((assistencias4$ASSISTENCIA - mean(assistencias4$ASSISTENCIA))^2))

print(paste("XGBoost RMSE:", xgb_rmse))
print(paste("XGBoost R-squared:", xgb_r2))

# Calcular RMSE e R-squared para Regressão Linear
lm_rmse <- sqrt(mean((lm_predictions - assistencias4$ASSISTENCIA)^2))
lm_r2 <- summary(lm_model)$r.squared

print(paste("Linear Regression RMSE:", lm_rmse))
print(paste("Linear Regression R-squared:", lm_r2))

# Combine as previsões dos modelos em um data frame para plotagem
results <- data.frame(
  Actual = assistencias4$ASSISTENCIA,
  RF_Predicted = rf_avg_predictions,
  XGB_Predicted = xgb_avg_predictions,
  LM_Predicted = lm_predictions
)

# Plotar os resultados usando ggplot2
ggplot(results, aes(x = Actual)) +
  geom_point(aes(y = RF_Predicted, color = "Random Forest"), alpha = 0.5) +
  geom_point(aes(y = XGB_Predicted, color = "XGBoost"), alpha = 0.5) +
  geom_point(aes(y = LM_Predicted, color = "Linear Regression"), alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "Assistência Real vs Prevista",
       x = "Assistência Real",
       y = "Assistência Prevista",
       color = "Modelo") +
  theme_minimal()


# Preparar dados para gráfico de métricas de performance
results_metrics <- data.frame(
  Model = c("Random Forest", "XGBoost", "Linear Regression"),
  RMSE = c(rf_rmse, xgb_rmse, lm_rmse),
  R_squared = c(rf_r2, xgb_r2, lm_r2)
)

# Transformar para formato longo para o gráfico
results_long <- pivot_longer(results_metrics, 
                             cols = c("RMSE", "R_squared"), 
                             names_to = "Metric", 
                             values_to = "Value")

# Plotar os resultados de métricas de performance
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  facet_wrap(~Metric, scales = "free_y") +  # Separar gráficos para RMSE e R-squared
  labs(title = "Métricas de performance dos Modelos",
       x = "Model",
       y = "Value") +
  scale_fill_manual(values = c("RMSE" = "blue", "R_squared" = "green")) +  # Cores para as métricas
  theme_minimal()



# Multiplicar apenas R_squared por 100 para representá-lo como porcentagem
results_long <- results_long %>%
  mutate(Value = ifelse(Metric == "R_squared", Value * 100, Value))  # Converte R_squared para porcentagem

# Plotar os resultados de métricas de performance
ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black", linewidth = 0.2) +  # Use linewidth em vez de size
  geom_text(aes(label = ifelse(Metric == "R_squared", paste0(round(Value, 2), "%"), round(Value, 2))), 
            position = position_dodge(0.9), 
            vjust = -0.5, 
            size = 5) +  # Adiciona rótulos acima das barras
  labs(title = "Performance Metrics for Different Models",
       x = "Model",
       y = "Value") +
  scale_fill_manual(values = c("RMSE" = "#0073C2FF", "R_squared" = "#EFC000FF")) +  # Cores para as métricas
  scale_y_continuous(labels = scales::number_format(), limits = c(0, max(results_long$Value) * 1.1)) +  # Ajusta limites do eixo y
  theme_minimal() +
  theme(
    legend.title = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),  # Centraliza e estiliza o título
    axis.text.x = element_text(angle = 45, hjust = 1)  # Rotaciona os rótulos do eixo x
  )

# métricas RMSE e R2 dos modelos todos
print(results_metrics)


# Criar uma tabela bonita com kableExtra
kable(results_metrics, format = "html", col.names = c("Modelo", "RMSE", "R-squared")) %>%
  kable_styling(full_width = FALSE, position = "left") %>%
  row_spec(0, bold = TRUE) %>%  # Destacar cabeçalho
  row_spec(1:nrow(results_metrics), background = "white") %>%  # Fundo branco para todas as linhas
  column_spec(1:3, color = "black")  # Texto preto para todas as células


# Exemplo de treinamento do modelo (ajuste conforme necessário)
rf_model <- train(ASSISTENCIA ~ ., data = assistencias4, method = "rf", trControl = trainControl(method = "cv"))


# Feature Importance para Random Forest
rf_importance <- importance(rf_model$finalModel)
rf_importance_df <- data.frame(Feature = rownames(rf_importance), Importance = rf_importance[, 1])

# Selecionar as 15 variáveis mais importantes
rf_importance_top15 <- rf_importance_df[order(-rf_importance_df$Importance), ][1:15, ]

# Normalizar a importância
rf_importance_top15$Importance <- rf_importance_top15$Importance / sum(rf_importance_top15$Importance) * 100  # Converte para porcentagem

# Plotar a importância das variáveis para Random Forest com valores normalizados
ggplot(rf_importance_top15, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Importance, 2)), 
            position = position_stack(vjust = 0.5), 
            color = "white") +  # Rótulos de porcentagem nas barras
  coord_flip() +
  labs(title = "15 Variáveis mais Importantes - Random Forest",
       x = "Variável",
       y = "Importância (%)") +
  theme_minimal()



#Exemplo de como criar a matriz (supondo que assistencias4 já está preparado)
assistencias4_matrix <- model.matrix(ASSISTENCIA ~ . - 1, data = assistencias4)  # Remove a interceptação
assistencias4_label <- assistencias4$ASSISTENCIA

# Criar o modelo XGBoost
xgb_model <- xgboost(data = assistencias4_matrix, label = assistencias4_label, 
                     nrounds = 100, objective = "reg:squarederror", verbose = 0)

# Feature Importance para XGBoost
xgb_importance <- xgb.importance(model = xgb_model)
xgb_importance_df <- as.data.frame(xgb_importance)

# Selecionar as 15 variáveis mais importantes
xgb_importance_top15 <- xgb_importance_df[order(-xgb_importance_df$Gain), ][1:15, ]

# Normalizar o ganho para porcentagem (opcional)
xgb_importance_top15$Gain <- xgb_importance_top15$Gain / sum(xgb_importance_top15$Gain) * 100  # Converte para porcentagem

# Plotar a importância das variáveis para XGBoost
ggplot(xgb_importance_top15, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = round(Gain, 2)), 
            position = position_stack(vjust = 0.5), 
            color = "white") +  # Rótulos de porcentagem nas barras
  coord_flip() +
  labs(title = "15 Variáveis mais Importantes - XGBoost",
       x = "Variável",
       y = "Ganho (%)") +
  theme_minimal()


# Criar dataframe com as variáveis usadas nos modelos 
variaveis <- data.frame(
  Nome = c("EPOCA", "CLASSIF", "VISITANTE", "CLASSIFVISIT", "JORNADA", 
           "TV", "ASSISTENCIA", "CHUVAJOGO", "CHUVA1HJOGO", "CHUVA2HJOGO", 
           "TEMPJOGO", "COBERTURAJOGO", "COBERTURA1HJOGO", "COBERTURA2HJOGO", 
           "GAMEBOXES", "JOGONUCLEOS", "MULHERESGARRA", "COVID", "DATA_YEAR", 
           "DATA_DAY", "HORAJOGO_HOUR", "HORAJOGO_MINUTE"),
  Descricao = c("Época do campeonato", "Classificação do time da casa", 
                "Nome do time visitante", "Classificação do time visitante", 
                "Número da jornada/rodada", "Canal de TV emissor", 
                "Número de espectadores", "Condição de chuva no jogo", 
                "Chuva 1 hora antes do jogo", "Chuva 2 horas antes do jogo", 
                "Temperatura durante o jogo", "Cobertura do céu no jogo", 
                "Cobertura 1 hora antes do jogo", "Cobertura 2 horas antes do jogo", 
                "Lugares anuais para sócios", 
                "Jogo direcionado aos núcleos", "Jogo direcionado às Mulheres", 
                "Indicador de COVID-19", "Ano do jogo", "Dia do jogo", 
                "Hora do jogo", "Minuto do jogo"),
  Tipo = c("Factor", "Numérica", "Factor", "Numérica", "Numérica", 
           "Factor", "Numérica", "Factor", "Factor", "Factor", 
           "Numérica", "Factor", "Factor", "Factor", "Numérica", 
           "Numérica", "Numérica", "Numérica", "Numérica", 
           "Inteira", "Inteira", "Inteira"),
  Fonte = rep("Pública", 22)
)

# Gerar a tabela formatada com kableExtra
kable(variaveis, format = "html", col.names = c("Nome da Variável", "Descrição", "Tipo da Variável", "Fonte")) %>%
  kable_styling(full_width = FALSE, position = "left") %>%
  row_spec(0, bold = TRUE) %>%  # Destacar cabeçalho
  row_spec(1:nrow(variaveis), background = "white") %>%  # Fundo branco para todas as linhas
  column_spec(1:4, color = "black")  # Texto preto para todas as células



# Criar dataframe com as variáveis iniciais
variaveis_assistencias <- data.frame(
  Nome = c("EPOCA", "CLASSIF", "PONTOS", "VISITANTE", "CLASSIFVISIT", 
           "PONTOSVISIT", "JORNADA", "PONTOSLIDER", "TV", 
           "ASSISTENCIA", "DATA", "HORAJOGO", "CHUVAJOGO", 
           "CHUVA1HJOGO", "CHUVA2HJOGO", "TEMPJOGO", "TEMP1HJOGO", 
           "COBERTURAJOGO", "COBERTURA1HJOGO", "COBERTURA2HJOGO", 
           "GAMEBOXES", "JOGONUCLEOS", "MULHERESGARRA", "COVID"),
  Descricao = c("Época do campeonato", "Classificação do time da casa", 
                "Número de pontos do time da casa", "Nome do time visitante", 
                "Classificação do time visitante", "Número de pontos do time visitante", 
                "Número da jornada/rodada", "Pontos do líder", 
                "Canal de TV emissor", "Número de espectadores", 
                "Data do jogo", "Hora do jogo", 
                "Condição de chuva no jogo", "Chuva 1 hora antes do jogo", 
                "Chuva 2 horas antes do jogo", "Temperatura durante o jogo", 
                "Temperatura 1 hora antes do jogo", 
                "Cobertura do céu no jogo", "Cobertura 1 hora antes do jogo", 
                "Cobertura 2 horas antes do jogo", "Número de lugares anuais", 
                "Jogo direcionado aos núcleos", "Jogo direcionado às Mulheres", 
                "Indicador de COVID-19"),
  Tipo = c("Factor", "Numérica", "Numérica", "Factor", "Numérica", 
           "Numérica", "Numérica", "Numérica", "Factor", 
           "Numérica", "Data", "POSIXct", "Factor", 
           "Factor", "Factor", "Numérica", "Numérica", 
           "Factor", "Factor", "Factor", "Numérica", 
           "Numérica", "Numérica", "Numérica"),
  Fonte = rep("Pública", 24)
)

# Gerar a tabela formatada com kableExtra
kable(variaveis_assistencias, format = "html", 
      col.names = c("Nome da Variável", "Descrição", "Tipo da Variável", "Fonte")) %>%
  kable_styling(full_width = FALSE, position = "left") %>%
  row_spec(0, bold = TRUE) %>%  # Destacar cabeçalho
  row_spec(1:nrow(variaveis_assistencias), background = "white") %>%  # Fundo branco para todas as linhas
  column_spec(1:4, color = "black") 

