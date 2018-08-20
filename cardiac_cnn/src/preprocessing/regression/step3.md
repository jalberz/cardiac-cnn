Predicting EF from Cardiac MRIs: Descriptive Analysis and Simple Linear Regression
================
Shashin Chokshi
July 16, 2018

Step 1: Ingest Data Processed by Python
---------------------------------------

``` r
rm(list=ls())
input_data = read.csv("~/Code/uchicago_bmi/clinical_informatics/ef_prediction/kaggle_ndsb2/train_validate_preprocessing/train_enriched.csv")
test_data = read.csv("~/Code/uchicago_bmi/clinical_informatics/ef_prediction/kaggle_ndsb2/test_enriched.csv")
```

Step 2: Load Key Libraries for Data Processing and Analysis
-----------------------------------------------------------

``` r
library(data.table)
library(Rmisc)
library(dplyr)
library(sqldf)
library(lubridate)
library(ggthemes)
library(ggplot2)
```

Step 3: Process Data and Create Table One Summary
-------------------------------------------------

TODO: Need to take out age

``` r
train_data <- input_data %>%
              filter(Id <= 500) %>%
              select('patient_id', 'rows', 'columns', 'spacing', 'slice_thickness', 'plane', 'slice_count', 'up_down_agg', 'age_years', 'sex',
                       'small_slice_count', 'normal_slice_count', 'angle', 'Systole', 'Diastole') %>%
              mutate(ejection_fraction = round((Diastole-Systole)/Diastole * 100,1)) 
```

    ## Warning: package 'bindrcpp' was built under R version 3.4.4

``` r
validate_data <- input_data %>%
                 filter(Id > 500) %>%
                select('patient_id', 'rows', 'columns', 'spacing', 'slice_thickness', 'plane', 'slice_count', 'up_down_agg', 'age_years', 'sex', 
                       'small_slice_count', 'normal_slice_count', 'angle', 'Systole', 'Diastole') %>%
                mutate(ejection_fraction = round((Diastole-Systole)/Diastole * 100,1)) 

test_data <- test_data %>%
                select('patient_id', 'rows', 'columns', 'spacing', 'slice_thickness', 'plane', 'slice_count', 'up_down_agg', 'age_years', 'sex', 
                       'small_slice_count', 'normal_slice_count', 'angle', 'Systole', 'Diastole') %>%
                mutate(ejection_fraction = round((Diastole-Systole)/Diastole * 100,1)) 


library(tableone)
```

    ## Warning: package 'tableone' was built under R version 3.4.4

``` r
dput(names(train_data))
```

    ## c("patient_id", "rows", "columns", "spacing", "slice_thickness", 
    ## "plane", "slice_count", "up_down_agg", "age_years", "sex", "small_slice_count", 
    ## "normal_slice_count", "angle", "Systole", "Diastole", "ejection_fraction"
    ## )

``` r
myVars = c("patient_id", "rows", "columns", "spacing", "slice_thickness", 
"plane", "slice_count", "up_down_agg", "age_years", "sex", "small_slice_count", 
"normal_slice_count", "angle", "Systole", "Diastole", "ejection_fraction")

catVars = c("sex", "plane")

tab1 <- CreateTableOne(vars = myVars, data = train_data, factorVars = catVars)
print(tab1, showAllLevels = TRUE)
```

    ##                                 
    ##                                  level Overall        
    ##   n                                       500         
    ##   patient_id (mean (sd))               250.50 (144.48)
    ##   rows (mean (sd))                     299.24 (107.76)
    ##   columns (mean (sd))                  253.92 (88.94) 
    ##   spacing (mean (sd))                    1.26 (0.34)  
    ##   slice_thickness (mean (sd))            7.75 (0.70)  
    ##   plane (%)                      COL       98 (19.6)  
    ##                                  ROW      402 (80.4)  
    ##   slice_count (mean (sd))               10.80 (1.82)  
    ##   up_down_agg (mean (sd))                9.18 (1.41)  
    ##   age_years (mean (sd))                 42.78 (20.40) 
    ##   sex (%)                        F        208 (41.6)  
    ##                                  M        292 (58.4)  
    ##   small_slice_count (mean (sd))          0.37 (1.33)  
    ##   normal_slice_count (mean (sd))        10.43 (1.19)  
    ##   angle (mean (sd))                     56.25 (10.02) 
    ##   Systole (mean (sd))                   71.96 (43.29) 
    ##   Diastole (mean (sd))                 165.87 (59.34) 
    ##   ejection_fraction (mean (sd))         58.46 (10.91)

Step 4: Visualize Jobs by Key Dimensions
----------------------------------------

``` r
# Systole Histogram
ggplot(train_data, aes(x=Systole)) + 
  geom_histogram(binwidth=5) +     
  labs(title="Systolic Volume", x = "End Systolic Volume (mLs)", y = "Number of Patients") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-1.png)

``` r
# Diastole Histogram
ggplot(train_data, aes(x=Diastole)) + 
  geom_histogram(binwidth=5) +     
  labs(title="Diastolic Volume", x = "End Diastolic Volume (mLs)", y = "Number of Patients") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-2.png)

``` r
# Slice Count Histogram
ggplot(train_data, aes(x=slice_count)) + 
  stat_count(binwidth=1) +     
  labs(title="Slice Count by Number of Patients", x = "Slice Count", y = "Number of Patients") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-3.png)

``` r
# Age vs. Sex
ggplot(train_data, aes(x=sex, y=age_years)) + 
  geom_boxplot() +     
  labs(title="Age vs Sex", x = "Sex", y = "Age (yrs)") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-4.png)

``` r
## Additional Graphs
ggplot(train_data, aes(x=age_years)) + 
  geom_histogram(binwidth=5) +     
  labs(title="Bandwidth by Count of Jobs", y = "Count of Jobs") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-5.png)

``` r
ggplot(train_data, aes(x=slice_thickness)) + 
  geom_histogram(binwidth=1) +     
  labs(title="Bandwidth by Count of Jobs", y = "Count of Jobs") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-6.png)

``` r
ggplot(train_data, aes(x=angle)) + 
  geom_histogram(binwidth=1) +     
  labs(title="Angle by Number of Patients", y = "Count of Jobs") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-7.png)

``` r
ggplot(train_data, aes(x=sex, y=age_years)) + 
  geom_boxplot() +     
  labs(title="Age for different Genders", x = "Age") + 
  theme_economist()
```

![](README_figs/README-unnamed-chunk-5-8.png)

Step 5: \#\# Step 5b: Run basic Linear Regression using DICOM metadata to Predict Values from Validate Dataset
--------------------------------------------------------------------------------------------------------------

``` r
### Systolic Regression
sys_reg <- lm(Systole ~ patient_id + rows + columns + spacing + slice_thickness + plane + slice_count + up_down_agg + age_years + sex + small_slice_count + normal_slice_count + angle, data=train_data)
summary(sys_reg)
```

    ## 
    ## Call:
    ## lm(formula = Systole ~ patient_id + rows + columns + spacing + 
    ##     slice_thickness + plane + slice_count + up_down_agg + age_years + 
    ##     sex + small_slice_count + normal_slice_count + angle, data = train_data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -118.59  -20.31   -4.99   13.71  316.74 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)        -2.513e+02  3.146e+01  -7.986 1.01e-14 ***
    ## patient_id         -3.244e-03  1.120e-02  -0.290  0.77226    
    ## rows                2.138e-02  6.628e-02   0.323  0.74715    
    ## columns             8.670e-02  7.411e-02   1.170  0.24259    
    ## spacing             6.338e+01  1.335e+01   4.749 2.69e-06 ***
    ## slice_thickness     3.159e+00  2.604e+00   1.213  0.22566    
    ## planeROW            4.778e+00  1.071e+01   0.446  0.65560    
    ## slice_count         1.273e+01  2.104e+00   6.052 2.86e-09 ***
    ## up_down_agg         4.670e+00  1.853e+00   2.521  0.01203 *  
    ## age_years           1.837e-01  1.188e-01   1.546  0.12280    
    ## sexM                9.065e+00  3.450e+00   2.628  0.00887 ** 
    ## small_slice_count  -1.167e+01  2.906e+00  -4.014 6.90e-05 ***
    ## normal_slice_count         NA         NA      NA       NA    
    ## angle              -3.499e-02  1.668e-01  -0.210  0.83391    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 35.74 on 487 degrees of freedom
    ## Multiple R-squared:  0.3349, Adjusted R-squared:  0.3185 
    ## F-statistic: 20.44 on 12 and 487 DF,  p-value: < 2.2e-16

``` r
# Create Predicted Value
validate_data$PredictedSystole <-predict(sys_reg, newdata = validate_data)


# Plot Predicted vs Actual Systoliv Volume
ggplot(validate_data, aes(x=validate_data$Systole, y=validate_data$PredictedSystole)) + 
  geom_point()+
  labs(title="Predicted vs Actual Systolic Volume", x = "Predicted Volume (mL)", y = "Actual Volume (mL)") + 
  geom_smooth(method=lm)
```

![](README_figs/README-unnamed-chunk-6-1.png)

``` r
# Calculate Key Metrics
cor(validate_data$Systole, validate_data$PredictedSystole)
```

    ## [1] 0.3494694

``` r
# RMSE 
sqrt(sum((validate_data$Systole-validate_data$PredictedSystole)^2))
```

    ## [1] 571.7247

``` r
sqrt(sum((sys_reg$fitted.values-train_data$Systole)^2))
```

    ## [1] 788.6695

``` r
### Diastolic Regression

dias_reg <- glm(Diastole ~ patient_id + rows + columns + spacing + slice_thickness + plane + slice_count + up_down_agg + age_years + sex + small_slice_count + normal_slice_count + angle, data=train_data)
summary(dias_reg)
```

    ## 
    ## Call:
    ## glm(formula = Diastole ~ patient_id + rows + columns + spacing + 
    ##     slice_thickness + plane + slice_count + up_down_agg + age_years + 
    ##     sex + small_slice_count + normal_slice_count + angle, data = train_data)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -174.595   -25.430    -3.159    21.206   263.417  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)        -456.16152   38.03247 -11.994  < 2e-16 ***
    ## patient_id           -0.01425    0.01354  -1.052  0.29320    
    ## rows                  0.03862    0.08012   0.482  0.63001    
    ## columns               0.24176    0.08958   2.699  0.00720 ** 
    ## spacing             127.57665   16.13239   7.908 1.77e-14 ***
    ## slice_thickness      15.06264    3.14770   4.785 2.27e-06 ***
    ## planeROW             18.56294   12.94150   1.434  0.15211    
    ## slice_count          17.23197    2.54302   6.776 3.57e-11 ***
    ## up_down_agg           6.26882    2.23955   2.799  0.00533 ** 
    ## age_years             0.01722    0.14361   0.120  0.90459    
    ## sexM                 21.03617    4.17022   5.044 6.43e-07 ***
    ## small_slice_count   -15.42324    3.51263  -4.391 1.39e-05 ***
    ## normal_slice_count         NA         NA      NA       NA    
    ## angle                 0.15683    0.20160   0.778  0.43699    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 1866.288)
    ## 
    ##     Null deviance: 1756896  on 499  degrees of freedom
    ## Residual deviance:  908882  on 487  degrees of freedom
    ## AIC: 5199.6
    ## 
    ## Number of Fisher Scoring iterations: 2

``` r
validate_data$PredictedDiastole <-predict(dias_reg, newdata = validate_data)

ggplot(validate_data, aes(x=validate_data$Diastole, y=validate_data$PredictedDiastole)) + 
  geom_point()+
  labs(title="Predicted vs Actual Diastolic Volume", x = "Predicted Volume (mL)", y = "Actual Volume (mL)") + 
  theme(plot.title = element_text(size=30)) + 
  geom_smooth(method=lm)
```

![](README_figs/README-unnamed-chunk-6-2.png)

``` r
cor(validate_data$Diastole, validate_data$PredictedDiastole)
```

    ## [1] 0.5385546

``` r
sqrt(sum((validate_data$Diastole-validate_data$PredictedDiastole)^2))
```

    ## [1] 736.4916

``` r
### Ejection Fraction
ef_reg <- glm(ejection_fraction ~ patient_id + rows + columns + spacing + slice_thickness + plane + slice_count + up_down_agg + age_years + sex + small_slice_count + normal_slice_count + angle, data=train_data)
summary(dias_reg)
```

    ## 
    ## Call:
    ## glm(formula = Diastole ~ patient_id + rows + columns + spacing + 
    ##     slice_thickness + plane + slice_count + up_down_agg + age_years + 
    ##     sex + small_slice_count + normal_slice_count + angle, data = train_data)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -174.595   -25.430    -3.159    21.206   263.417  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)        -456.16152   38.03247 -11.994  < 2e-16 ***
    ## patient_id           -0.01425    0.01354  -1.052  0.29320    
    ## rows                  0.03862    0.08012   0.482  0.63001    
    ## columns               0.24176    0.08958   2.699  0.00720 ** 
    ## spacing             127.57665   16.13239   7.908 1.77e-14 ***
    ## slice_thickness      15.06264    3.14770   4.785 2.27e-06 ***
    ## planeROW             18.56294   12.94150   1.434  0.15211    
    ## slice_count          17.23197    2.54302   6.776 3.57e-11 ***
    ## up_down_agg           6.26882    2.23955   2.799  0.00533 ** 
    ## age_years             0.01722    0.14361   0.120  0.90459    
    ## sexM                 21.03617    4.17022   5.044 6.43e-07 ***
    ## small_slice_count   -15.42324    3.51263  -4.391 1.39e-05 ***
    ## normal_slice_count         NA         NA      NA       NA    
    ## angle                 0.15683    0.20160   0.778  0.43699    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 1866.288)
    ## 
    ##     Null deviance: 1756896  on 499  degrees of freedom
    ## Residual deviance:  908882  on 487  degrees of freedom
    ## AIC: 5199.6
    ## 
    ## Number of Fisher Scoring iterations: 2

``` r
validate_data$pred_ejection_fraction <-predict(ef_reg, newdata = validate_data)

ggplot(validate_data, aes(x=validate_data$ejection_fraction, y=validate_data$pred_ejection_fraction)) + 
  geom_point()+
  labs(title="Predicted vs Actual Ejection Fraction", x = "Predicted Ejection Fraction (%)", y = "Actual Ejection Fraction (%)") + 
  theme(plot.title = element_text(size=30)) + 
  geom_smooth(method=lm)
```

![](README_figs/README-unnamed-chunk-6-3.png)

``` r
cor(validate_data$ejection_fraction, validate_data$pred_ejection_fraction)
```

    ## [1] 0.0745924

``` r
sqrt(sum((validate_data$ejection_fraction-validate_data$pred_ejection_fraction)^2))
```

    ## [1] 163.8356

Step 5b: Run basic Linear Regression using DICOM metadata to Predict Values from Test Dataset
---------------------------------------------------------------------------------------------

``` r
### Systolic Regression
sys_reg <- lm(Systole ~ patient_id + rows + columns + spacing + slice_thickness + plane + slice_count + up_down_agg + age_years + sex + small_slice_count + normal_slice_count + angle, data=train_data)
summary(sys_reg)
```

    ## 
    ## Call:
    ## lm(formula = Systole ~ patient_id + rows + columns + spacing + 
    ##     slice_thickness + plane + slice_count + up_down_agg + age_years + 
    ##     sex + small_slice_count + normal_slice_count + angle, data = train_data)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -118.59  -20.31   -4.99   13.71  316.74 
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)        -2.513e+02  3.146e+01  -7.986 1.01e-14 ***
    ## patient_id         -3.244e-03  1.120e-02  -0.290  0.77226    
    ## rows                2.138e-02  6.628e-02   0.323  0.74715    
    ## columns             8.670e-02  7.411e-02   1.170  0.24259    
    ## spacing             6.338e+01  1.335e+01   4.749 2.69e-06 ***
    ## slice_thickness     3.159e+00  2.604e+00   1.213  0.22566    
    ## planeROW            4.778e+00  1.071e+01   0.446  0.65560    
    ## slice_count         1.273e+01  2.104e+00   6.052 2.86e-09 ***
    ## up_down_agg         4.670e+00  1.853e+00   2.521  0.01203 *  
    ## age_years           1.837e-01  1.188e-01   1.546  0.12280    
    ## sexM                9.065e+00  3.450e+00   2.628  0.00887 ** 
    ## small_slice_count  -1.167e+01  2.906e+00  -4.014 6.90e-05 ***
    ## normal_slice_count         NA         NA      NA       NA    
    ## angle              -3.499e-02  1.668e-01  -0.210  0.83391    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 35.74 on 487 degrees of freedom
    ## Multiple R-squared:  0.3349, Adjusted R-squared:  0.3185 
    ## F-statistic: 20.44 on 12 and 487 DF,  p-value: < 2.2e-16

``` r
# Create Predicted Value
test_data$PredictedSystole <-predict(sys_reg, newdata = test_data)


# Plot Predicted vs Actual Systoliv Volume
ggplot(test_data, aes(x=test_data$Systole, y=test_data$PredictedSystole)) + 
  geom_point()+
  theme(plot.title = element_text(size=30)) + 
  labs(title="Predicted vs Actual Systolic Volume", x = "Predicted Volume (mL)", y = "Actual Volume (mL)") + 
  geom_smooth(method=lm)
```

![](README_figs/README-unnamed-chunk-7-1.png)

``` r
# Calculate Key Metrics
cor(test_data$Systole, test_data$PredictedSystole)
```

    ## [1] 0.5797133

``` r
# RMSE 
sqrt(sum((test_data$Systole-test_data$PredictedSystole)^2))
```

    ## [1] 654.8027

``` r
sqrt(sum((sys_reg$fitted.values-train_data$Systole)^2))
```

    ## [1] 788.6695

``` r
### Diastolic Regression

dias_reg <- glm(Diastole ~ patient_id + rows + columns + spacing + slice_thickness + plane + slice_count + up_down_agg + age_years + sex + small_slice_count + normal_slice_count + angle, data=train_data)
summary(dias_reg)
```

    ## 
    ## Call:
    ## glm(formula = Diastole ~ patient_id + rows + columns + spacing + 
    ##     slice_thickness + plane + slice_count + up_down_agg + age_years + 
    ##     sex + small_slice_count + normal_slice_count + angle, data = train_data)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -174.595   -25.430    -3.159    21.206   263.417  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)        -456.16152   38.03247 -11.994  < 2e-16 ***
    ## patient_id           -0.01425    0.01354  -1.052  0.29320    
    ## rows                  0.03862    0.08012   0.482  0.63001    
    ## columns               0.24176    0.08958   2.699  0.00720 ** 
    ## spacing             127.57665   16.13239   7.908 1.77e-14 ***
    ## slice_thickness      15.06264    3.14770   4.785 2.27e-06 ***
    ## planeROW             18.56294   12.94150   1.434  0.15211    
    ## slice_count          17.23197    2.54302   6.776 3.57e-11 ***
    ## up_down_agg           6.26882    2.23955   2.799  0.00533 ** 
    ## age_years             0.01722    0.14361   0.120  0.90459    
    ## sexM                 21.03617    4.17022   5.044 6.43e-07 ***
    ## small_slice_count   -15.42324    3.51263  -4.391 1.39e-05 ***
    ## normal_slice_count         NA         NA      NA       NA    
    ## angle                 0.15683    0.20160   0.778  0.43699    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 1866.288)
    ## 
    ##     Null deviance: 1756896  on 499  degrees of freedom
    ## Residual deviance:  908882  on 487  degrees of freedom
    ## AIC: 5199.6
    ## 
    ## Number of Fisher Scoring iterations: 2

``` r
test_data$PredictedDiastole <-predict(dias_reg, newdata = test_data)

ggplot(test_data, aes(x=test_data$Diastole, y=test_data$PredictedDiastole)) + 
  geom_point()+
  labs(title="Predicted vs Actual Diastolic Volume", x = "Predicted Volume (mL)", y = "Actual Volume (mL)") + 
  theme(plot.title = element_text(size=30)) + 
  geom_smooth(method=lm)
```

![](README_figs/README-unnamed-chunk-7-2.png)

``` r
cor(test_data$Diastole, test_data$PredictedDiastole)
```

    ## [1] 0.7052103

``` r
sqrt(sum((test_data$Diastole-test_data$PredictedDiastole)^2))
```

    ## [1] 881.9993

``` r
### Ejection Fraction
ef_reg <- glm(ejection_fraction ~ patient_id + rows + columns + spacing + slice_thickness + plane + slice_count + up_down_agg + age_years + sex + small_slice_count + normal_slice_count + angle, data=train_data)
summary(dias_reg)
```

    ## 
    ## Call:
    ## glm(formula = Diastole ~ patient_id + rows + columns + spacing + 
    ##     slice_thickness + plane + slice_count + up_down_agg + age_years + 
    ##     sex + small_slice_count + normal_slice_count + angle, data = train_data)
    ## 
    ## Deviance Residuals: 
    ##      Min        1Q    Median        3Q       Max  
    ## -174.595   -25.430    -3.159    21.206   263.417  
    ## 
    ## Coefficients: (1 not defined because of singularities)
    ##                      Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)        -456.16152   38.03247 -11.994  < 2e-16 ***
    ## patient_id           -0.01425    0.01354  -1.052  0.29320    
    ## rows                  0.03862    0.08012   0.482  0.63001    
    ## columns               0.24176    0.08958   2.699  0.00720 ** 
    ## spacing             127.57665   16.13239   7.908 1.77e-14 ***
    ## slice_thickness      15.06264    3.14770   4.785 2.27e-06 ***
    ## planeROW             18.56294   12.94150   1.434  0.15211    
    ## slice_count          17.23197    2.54302   6.776 3.57e-11 ***
    ## up_down_agg           6.26882    2.23955   2.799  0.00533 ** 
    ## age_years             0.01722    0.14361   0.120  0.90459    
    ## sexM                 21.03617    4.17022   5.044 6.43e-07 ***
    ## small_slice_count   -15.42324    3.51263  -4.391 1.39e-05 ***
    ## normal_slice_count         NA         NA      NA       NA    
    ## angle                 0.15683    0.20160   0.778  0.43699    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for gaussian family taken to be 1866.288)
    ## 
    ##     Null deviance: 1756896  on 499  degrees of freedom
    ## Residual deviance:  908882  on 487  degrees of freedom
    ## AIC: 5199.6
    ## 
    ## Number of Fisher Scoring iterations: 2

``` r
test_data$pred_ejection_fraction <-predict(ef_reg, newdata = test_data)

ggplot(test_data, aes(x=test_data$ejection_fraction, y=test_data$pred_ejection_fraction)) + 
  geom_point()+
  labs(title="Predicted vs Actual Ejection Fraction", x = "Predicted Ejection Fraction (%)", y = "Actual Ejection Fraction (%)") + 
  theme(plot.title = element_text(size=30)) + 
  geom_smooth(method=lm)
```

![](README_figs/README-unnamed-chunk-7-3.png)

``` r
cor(test_data$ejection_fraction, test_data$pred_ejection_fraction)
```

    ## [1] 0.2995586

``` r
sqrt(sum((test_data$ejection_fraction-test_data$pred_ejection_fraction)^2))
```

    ## [1] 194.6722

Step 5: Calculate Key Metrics
-----------------------------
