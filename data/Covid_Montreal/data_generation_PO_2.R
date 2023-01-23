
##########################################################################################
# Create potential outcome for simulation based on metadata_clean.csv
# 
# ignorability does not hold
##########################################################################################

##########################################################################################
# Input: metadata_clean.csv
# Data source: https://github.com/ieee8023/covid-chestxray-dataset
# Paper source:  Cohen JP, Morrison P, Dao L (2020) Covid-19 image data collection. arXiv:2003.11597
# Cohen JP, Morrison P, Dao L, Roth K, Duong TQ, Ghassemi M (2020) Covid-19 image data collection:
# Prospective predictions are the future. arXiv:2006.11988

# Output: metadata_PO_x.csv
# author : XXX
# Date: Dec-27-2022
##########################################################################################



for (i in seq(from=0.1, to=0.5, by=0.1) ) {

dat<-read.csv('metadata_clean_for_PO.csv')

attach(dat,warn.conflicts = FALSE)

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


## Create treatment depending on X, ignorability does not hold
attach(dat,warn.conflicts = FALSE)

XX<-2*age+3*sex+5*Pneumonia_Viral+4*Pneumonia_Bacterial+
  (-1)*Pneumonia_Fungal+Pneumonia_Other-2*Pneumonia_Unknown+
  8*Tuberculosis

## standardize XX
XXX<-(XX-mean(XX))/sd(XX)

## Logistic Sigmoid Function
pp<-(1+exp(-XXX))^-1

## random generate treatment
dat$treatment<-rbinom(dim(dat)[1],1,(pp*i)/mean(pp) )

##

# summary(pp)
# 
# summary(dat$treatment)
# 


attach(dat,warn.conflicts = FALSE)

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
  btt0*log(age)*Pneumonia_Viral+btt1*log(age)*Pneumonia_Bacterial+
  btt2*log(age)*Pneumonia_Fungal+btt3*log(age)*Pneumonia_Other+
  btt4*log(age)*Pneumonia_Unknown+btt5*log(age)*sex*Tuberculosis+
    e1

##



write.csv(dat,file=paste( paste( "metadata_PO",i, sep="_"),"x.csv", sep=""),
          row.names = FALSE)



}











