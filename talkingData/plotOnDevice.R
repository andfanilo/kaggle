library(ggplot2)

## TO RUN AFTER simplePredictOnDevice.R

# let's try and plot some things
pl <- ggplot(data = train_device, aes(x = phone_brand)) 
pl + geom_bar(fill = 'lightblue', colour = 'black')