# Project pattern

In case this project would be further developed in the future, there will be a separation of concerns applied. **MVC** design pattern will be used for clarity, maintainability and scalability.  

!!! Term *model* is used here in two different contexts. 'model' as machine learning algorithm and 'Model' as an element of MVC pattern.
To prevent confusion the MVC's Model will be called further ***Logic***.

## Logic
The Model handles the core logic of the application. This includes
- data processing
- feature extraction
- model training/prediction
- model's comparison

## View
The View is responsible for 

- plots
- presenting data
- generating reports 

## Controller
The Controller manages

- managing workflow
- handling user inputs and actions
- coordinating between Logic and View