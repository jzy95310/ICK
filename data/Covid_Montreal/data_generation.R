
##########################################################################################
# Data cleaning for metadata.csv
# recode character to numeric
# impute missing values
##########################################################################################

##########################################################################################
# Input: metadata.csv
# Data source: https://github.com/ieee8023/covid-chestxray-dataset
# Paper source:  Cohen JP, Morrison P, Dao L (2020) Covid-19 image data collection. arXiv:2003.11597
# Cohen JP, Morrison P, Dao L, Roth K, Duong TQ, Ghassemi M (2020) Covid-19 image data collection:
# Prospective predictions are the future. arXiv:2006.11988
# Output: metadata_clean.csv
# author : XXX
# Date: Dec-01-2022
##########################################################################################

library(tidyverse)

set.seed(1234)

## data cleaning 

dat<-read.csv('metadata.csv')[,1:24]



## patientid----

# new ID start from 482

dat$patientid[dat$patientid=='311a']<-482

dat$patientid[dat$patientid=='311b']<-483


dat$patientid[dat$patientid=='313a']<-484

dat$patientid[dat$patientid=='313b']<-485


dat$patientid[dat$patientid=='319a']<-486
dat$patientid[dat$patientid=='319b']<-486

dat$patientid[dat$patientid=='324a']<-487

dat$patientid[dat$patientid=='324b']<-488


dat$patientid[dat$patientid=='326a']<-489


dat$patientid[dat$patientid=='326b']<-490


dat$patientid[dat$patientid=='331a']<-491


## same people different time point and different measure
dat$patientid[dat$patientid=='331b' & dat$offset==10]<-492
dat$patientid[dat$patientid=='331b' & dat$offset==13]<-493

## same people different time point and different measure
dat$patientid[dat$patientid=='350a' & dat$offset==7]<-494
dat$patientid[dat$patientid=='350a' & dat$offset==15]<-495


dat$patientid[dat$patientid=='350b']<-496
dat$patientid[dat$patientid=='350c']<-497

dat$patientid[dat$patientid=='360a']<-498

dat$patientid[dat$patientid=='360b']<-499

dat$patientid[dat$patientid=='361a']<-500

dat$patientid[dat$patientid=='361b']<-501

dat$patientid[dat$patientid=='423a']<-502
dat$patientid[dat$patientid=='423b']<-503

dat$patientid[dat$patientid=='425c']<-504

dat$patientid[dat$patientid=='425d']<-505


dat$patientid[dat$patientid=='426a']<-506

dat$patientid[dat$patientid=='426b']<-507


dat$patientid[dat$patientid=='427a']<-508

dat$patientid[dat$patientid=='427b']<-509


dat$patientid<-as.numeric(dat$patientid)



##sex----

dat$sex[dat$sex=='M']<-0

dat$sex[dat$sex=='F']<-1

dat$sex<-as.numeric(dat$sex)

## RT_PCR_positive----

dat$RT_PCR_positive[dat$RT_PCR_positive =='Y']<-1

dat$RT_PCR_positive[dat$RT_PCR_positive =='Unclear']<-0

dat$RT_PCR_positive[dat$RT_PCR_positive =='']<-NA

dat$RT_PCR_positive<-as.numeric(dat$RT_PCR_positive)

## 

## create function for recoding similar data type
recode<-function(X){
  
  X[X =='Y']<-1
  X[X =='N']<-0
  X[X =='']<-NA
return(as.numeric(X))

}

dat$survival<-recode(dat$survival)

dat$intubated<-recode(dat$intubated)

dat$intubation_present<-recode(dat$intubation_present)

dat$went_icu<-recode(dat$went_icu)

dat$in_icu<-recode(dat$in_icu)

dat$needed_supplemental_O2<-recode(dat$needed_supplemental_O2)

dat$extubated<-recode(dat$extubated)

## modality----
## delete CT image: 84
dat<-dat[dat$modality=='X-ray',]

## view----
dat$view<-as.numeric(factor(dat$view))

## location----
## code NA as 1 
dat$location<-as.numeric(factor(dat$location))


## delete columns with large proportion of missingness: 
## needed_supplemental_O2; extubated; temperature;pO2_saturation;
## leukocyte_count; neutrophil_count; lymphocyte_count

## delete columns not related to disease or image: 
## modality; date; folder

dat<-dat[,c(1:11,19,22,24)]



### Imputation for missing data


## offset
## create function for random sampling from 25%-75% to impute
rand_impute_cont<-function(dd) {
  
  p1<-unname(quantile(dd, probs =0.25, na.rm = TRUE ))
  
  p2<-unname(quantile(dd, probs =0.75, na.rm = TRUE ))
  
  pool<-dd[dd<=p2 & dd>=p1 & !is.na(dd) ]
  
  
  for(i in 1:length(dd)) {
    if(is.na(dd[i])) {
      dd[i] <- sample(pool,1)
      
    } 
  }
  return(dd)
  
}

dat$offset<-rand_impute_cont(dat$offset)


## sex

## function for random imputing with bernoulli with p=proportion
rand_impute<-function(dd) {
  
  p<-mean(dd,na.rm = TRUE)

  for(i in 1:length(dd)) {
    if(is.na(dd[i])) {
      dd[i] <- rbinom(1,1,p)
  
    } 
  }
  return(dd)

}

dat$sex<-rand_impute(dat$sex)


## age
dat$age<-rand_impute_cont(dat$age)


## RT_PCR_positive 
## impute based on findings, if COVID then RT_PCR_positive=1
for(i in 1:length(dat$RT_PCR_positive)) {
  if(is.na(dat$RT_PCR_positive[i])) {
    
    dat$RT_PCR_positive[i] <- ifelse(dat$finding[i]=="Pneumonia/Viral/COVID-19",
                                     1,0)
    
  } 
}



## survival
dat$survival<-rand_impute(dat$survival)

## intubated
dat$intubated<-rand_impute(dat$intubated)

##  intubation_present
dat$intubation_present<-rand_impute(dat$intubation_present)

## went_icu
dat$went_icu<-rand_impute(dat$went_icu)

## in_icu
dat$in_icu<-rand_impute(dat$in_icu)


## finding----

dat<-dat %>% relocate(finding, .after = last_col())

## reserve the original finding variable in their character form
dat$finding_0<-factor(dat$finding)


## recode findings for generate potential outcome later

#levels(factor(dat$finding))


## recode findings according to type: 
dat$finding[dat$finding=='Pneumonia/Viral/COVID-19']<-1

dat$finding[dat$finding=='Pneumonia/Viral/Herpes ']<-1

dat$finding[dat$finding=='Pneumonia/Viral/Influenza']<-1

dat$finding[dat$finding=='Pneumonia/Viral/Influenza/H1N1']<-1

dat$finding[dat$finding=='Pneumonia/Viral/MERS-CoV']<-1

dat$finding[dat$finding=='Pneumonia/Viral/SARS']<-1

dat$finding[dat$finding=='Pneumonia/Viral/Varicella']<-1

dat$finding[dat$finding=='Pneumonia/Bacterial']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/Chlamydophila']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/E.Coli']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/Klebsiella']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/Legionella']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/Mycoplasma']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/Nocardia']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/Staphylococcus/MRSA']<-2

dat$finding[dat$finding=='Pneumonia/Bacterial/Streptococcus']<-2

dat$finding[dat$finding=='Pneumonia/Fungal/Aspergillosis']<-3

dat$finding[dat$finding=='Pneumonia/Fungal/Pneumocystis']<-3

dat$finding[dat$finding=='Pneumonia/Lipoid']<-4

dat$finding[dat$finding=='Pneumonia/Aspiration']<-4

dat$finding[dat$finding=='Pneumonia']<-5

dat$finding[dat$finding=='Tuberculosis']<-6

dat$finding[dat$finding=='Unknown']<-5

dat$finding[dat$finding=='todo']<-5

dat$finding[dat$finding=='No Finding']<-7

dat$finding<-as.numeric(dat$finding)

## one hot encoding

dat <- mutate(dat, Pneumonia_Viral = ifelse(finding==1,1,0))

dat <- mutate(dat, Pneumonia_Bacterial = ifelse(finding==2,1,0))

dat <- mutate(dat, Pneumonia_Fungal = ifelse(finding==3,1,0))

dat <- mutate(dat, Pneumonia_Other = ifelse(finding==4,1,0))

dat <- mutate(dat, Pneumonia_Unknown = ifelse(finding==5,1,0))

dat <- mutate(dat, Tuberculosis = ifelse(finding==6,1,0))

## delete extra finding col

dat<-subset(dat, select= -c(finding))

## offset----
## delete some very small offset data
dat<-dat[dat$offset>(-30) & dat$offset<200,]


## write data to csv

write.csv(dat,'metadata_clean_for_PO.csv',row.names = FALSE)












