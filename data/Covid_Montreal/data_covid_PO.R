
##########################################################################################
# Create potential outcome for simulation based on metadata_clean.csv
# 
# 
##########################################################################################

##########################################################################################
# Input: metadata_clean.csv
# Data source: https://github.com/ieee8023/covid-chestxray-dataset
# Paper source:  Cohen JP, Morrison P, Dao L (2020) Covid-19 image data collection. arXiv:2003.11597
# Cohen JP, Morrison P, Dao L, Roth K, Duong TQ, Ghassemi M (2020) Covid-19 image data collection:
# Prospective predictions are the future. arXiv:2006.11988

# Output: metadata_PO.csv
# author : Zhuoran Hou
# Date: Dec-01-2022
##########################################################################################



dat<-read.csv('metadata_clean_for_PO.csv')

attach(dat)

set.seed(1234)

#summary(dat)

### set up coef

b0<-2/100

b1<-1/100

b2<-1.5/100

b3<-4/100

b4<-2/100

b5<-(-5)/100

b6<-4/100

b7<-5/100

b8<-6/100

b9<-7/100

##
b16<-3/100

b17<-2/100

b18<-2.5/100

b19<-4/100

#Pneumonia_Viral
b10<-4/100
bt0<-(-50)/100
btt0<-6/100

#Pneumonia_Bacterial    
b11<-3/100
bt1<-(-70)/100
btt1<-1/100

#Pneumonia_Fungal  
b12<-6/100
bt2<-(-40)/100
btt2<-3/100

# Pneumonia_Other  
b13<-2/100
bt3<-(-60)/100
btt3<-3/100

#Pneumonia_Unknown   
b14<-3.5/100
bt4<-(-65)/100
btt4<-2/100

# Tuberculosis  
b15<-7/100
bt5<-(-80)/100
btt5<-(-2/100)

# treatment
#bt<-(-80)/100

## set up other parameters
e0<-rnorm(dim(dat)[1],mean = 0, sd =0.1)
e1<-rnorm(dim(dat)[1],mean = 0, sd =0.1)


## random generate treatment
dat$treatment<-rbinom(dim(dat)[1],1,0.5)

attach(dat)

#### Y0

dat$Y0<-b0+b1*offset+b2*sex+b3*age+b4*RT_PCR_positive+b5*survival+
  b6*intubated+b7*intubation_present+b8*went_icu+b9*in_icu+
  b10*Pneumonia_Viral+b11*Pneumonia_Bacterial+
  b12*Pneumonia_Fungal+b13*Pneumonia_Other+
  b14*Pneumonia_Unknown+b15*Tuberculosis+
  b16*age*RT_PCR_positive+b17*age*sex+b18*went_icu*age*sex+
  b19*age*intubation_present+
  e0


#### Y1

dat$Y1<-b0+b1*offset+b2*sex+b3*age+b4*RT_PCR_positive+b5*survival+
  b6*intubated+b7*intubation_present+b8*went_icu+b9*in_icu+
  b10*Pneumonia_Viral+b11*Pneumonia_Bacterial+
  b12*Pneumonia_Fungal+b13*Pneumonia_Other+
  b14*Pneumonia_Unknown+b15*Tuberculosis+
  b16*age*RT_PCR_positive+b17*age*sex+b18*went_icu*age*sex+
  b19*age*intubation_present+
  bt0*Pneumonia_Viral+bt1*Pneumonia_Bacterial+
  bt2*Pneumonia_Fungal+bt3*Pneumonia_Other+
  bt4*Pneumonia_Unknown+bt5*Tuberculosis+
  btt0*age*Pneumonia_Viral+btt1*age*Pneumonia_Bacterial+
  btt2*age*Pneumonia_Fungal+btt3*age*Pneumonia_Other+
  btt4*age*Pneumonia_Unknown+btt5*age*sex*Tuberculosis+
    e1



summary(dat)


write.csv(dat,'metadata_PO.csv',row.names = FALSE)


head(dat$Y1-dat$Y0)


hist(dat$Y0)

hist(dat$Y1)


hist(dat$Y1-dat$Y0)









