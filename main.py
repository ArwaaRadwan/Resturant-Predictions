import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf

# Load the model and input features
model = joblib.load("Model_test.pk1")
input_features = joblib.load("input.pk1")

# Define a prediction function
def prediction(online_order, book_table, rest_type, location, approx_cost, listed_in_type, listed_in_city):
    df = pd.DataFrame(columns=input_features)
    df.at[0, "online_order"] = online_order
    df.at[0, "book_table"] = book_table
    df.at[0, "rest_type"] = rest_type
    df.at[0, "location"] = location
    df.at[0, "approx_cost(for two people)"] = approx_cost
    df.at[0, "listed_in(type)"] = listed_in_type
    df.at[0, "listed_in(city)"] = listed_in_city
    
    result = model.predict(df)
    return result[0]

# Streamlit app
st.title('Restaurant Success Prediction')

st.sidebar.header('User Input Features')

def user_input_features():
    online_order = st.sidebar.selectbox('Online Order', ['Yes', 'No'])
    book_table = st.sidebar.selectbox('Book Table', ['Yes', 'No'])
    rest_type = st.sidebar.selectbox('Restaurant Type', ['Casual Dining', 'Cafe', 'Quick Bites',
       'Casual Dining, Cafe', 'Quick Bites, Cafe', 'Cafe, Quick Bites', 'Delivery', 'Mess', 
       'Dessert Parlor', 'Bakery, Dessert Parlor', 'Pub', 'Bakery', 'Takeaway, Delivery', 'Fine Dining', 
       'Beverage Shop', 'Sweet Shop', 'Bar', 'Beverage Shop, Quick Bites', 'Confectionery',
       'Quick Bites, Beverage Shop', 'Dessert Parlor, Sweet Shop', 'Bakery, Quick Bites', 'Sweet Shop, Quick Bites',
       'Kiosk', 'Food Truck', 'Quick Bites, Dessert Parlor', 'Beverage Shop, Dessert Parlor', 'Takeaway', 
       'Pub, Casual Dining', 'Casual Dining, Bar', 'Dessert Parlor, Beverage Shop', 'Quick Bites, Bakery', 
       'Dessert Parlor, Quick Bites', 'Microbrewery, Casual Dining', 'Lounge', 'Bar, Casual Dining', 'Food Court',
       'Cafe, Bakery', 'Dhaba', 'Quick Bites, Sweet Shop', 'Microbrewery', 'Food Court, Quick Bites', 'Pub, Bar',
       'Casual Dining, Pub', 'Lounge, Bar', 'Food Court, Dessert Parlor', 'Casual Dining, Sweet Shop', 
       'Food Court, Casual Dining', 'Casual Dining, Microbrewery', 'Sweet Shop, Dessert Parlor', 'Bakery, Beverage Shop',
       'Lounge, Casual Dining', 'Cafe, Food Court', 'Beverage Shop, Cafe', 'Cafe, Dessert Parlor', 'Dessert Parlor, Cafe',
       'Dessert Parlor, Bakery', 'Microbrewery, Pub', 'Bakery, Food Court', 'Club', 'Quick Bites, Food Court', 
       'Bakery, Cafe', 'Bar, Cafe', 'Pub, Cafe', 'Casual Dining, Irani Cafee', 'Fine Dining, Lounge', 'Bar, Quick Bites',
       'Bakery, Kiosk', 'Pub, Microbrewery', 'Cafe, Lounge', 'Bar, Pub', 'Lounge, Cafe', 'Club, Casual Dining', 
       'Quick Bites, Mess', 'Quick Bites, Meat Shop', 'Quick Bites, Kiosk', 'Lounge, Microbrewery', 'Food Court, Beverage Shop',
       'Dessert Parlor, Food Court', 'Bar, Lounge'])
    location = st.sidebar.selectbox('Location', ['Banashankari', 'Basavanagudi', 'Mysore Road', 'Jayanagar',
       'Kumaraswamy Layout', 'Rajarajeshwari Nagar', 'Vijay Nagar', 'Uttarahalli', 'JP Nagar', 'South Bangalore', 
       'City Market', 'Nagarbhavi', 'Bannerghatta Road', 'BTM', 'Kanakapura Road', 'Bommanahalli', 'CV Raman Nagar', 
       'Electronic City', 'HSR', 'Marathahalli', 'Sarjapur Road', 'Wilson Garden', 'Shanti Nagar', 'Koramangala 5th Block', 
       'Koramangala 8th Block', 'Richmond Road', 'Koramangala 7th Block', 'Jalahalli', 'Koramangala 4th Block', 'Bellandur',
       'Whitefield', 'East Bangalore', 'Old Airport Road', 'Indiranagar', 'Koramangala 1st Block', 'Frazer Town', 'RT Nagar', 
       'MG Road', 'Brigade Road', 'Lavelle Road', 'Church Street', 'Ulsoor', 'Residency Road', 'Shivajinagar', 'Infantry Road',
       'St. Marks Road', 'Cunningham Road', 'Race Course Road', 'Commercial Street', 'Vasanth Nagar', 'HBR Layout', 'Domlur', 
       'Ejipura', 'Jeevan Bhima Nagar', 'Old Madras Road', 'Malleshwaram', 'Seshadripuram', 'Kammanahalli', 'Koramangala 6th Block',
       'Majestic', 'Langford Town', 'Central Bangalore', 'Sanjay Nagar', 'Brookefield', 'ITPL Main Road, Whitefield', 
       'Varthur Main Road, Whitefield', 'KR Puram', 'Koramangala 2nd Block', 'Koramangala 3rd Block', 'Koramangala', 
       'Hosur Road', 'Rajajinagar', 'Banaswadi', 'North Bangalore', 'Nagawara', 'Hennur', 'Kalyan Nagar', 'New BEL Road', 
       'Jakkur', 'Rammurthy Nagar', 'Thippasandra', 'Kaggadasapura', 'Hebbal', 'Kengeri', 'Sankey Road', 'Sadashiv Nagar', 
       'Basaveshwara Nagar', 'Yeshwantpur', 'West Bangalore', 'Magadi Road', 'Yelahanka', 'Sahakara Nagar', 'Peenya'])
    listed_in_type = st.sidebar.selectbox('Restaurant Type', ['Buffet', 'Cafes', 'Delivery', 'Desserts', 'Dine-out',
       'Drinks & nightlife', 'Pubs and bars'])
    listed_in_city = st.sidebar.selectbox('Listed In (City)', ['Banashankari', 'Bannerghatta Road', 'Basavanagudi', 'Bellandur',
       'Brigade Road', 'Brookefield', 'BTM', 'Church Street', 'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar',
       'Jayanagar', 'JP Nagar', 'Kalyan Nagar', 'Kammanahalli', 'Koramangala 4th Block', 'Koramangala 5th Block', 
       'Koramangala 6th Block', 'Koramangala 7th Block', 'Lavelle Road', 'Malleshwaram', 'Marathahalli', 'MG Road', 
       'New BEL Road', 'Old Airport Road', 'Rajajinagar', 'Residency Road', 'Sarjapur Road', 'Whitefield'])
    approx_cost = st.sidebar.slider('Approx Cost (for two people)', min_value=10, max_value=3500, step=1, value=500)
    
    data = {
        'online_order': online_order,
        'book_table': book_table,
        'rest_type': rest_type,
        'location': location,
        'approx_cost(for two people)': approx_cost,
        'listed_in(type)': listed_in_type,
        'listed_in(city)': listed_in_city,
    }
    return data

# Get user input
input_data = user_input_features()
input_df = pd.DataFrame([input_data])

st.subheader('User Input Parameters')
st.write(input_df)

# Make prediction
if st.button('Predict'):
    result = prediction(
        input_data['online_order'],
        input_data['book_table'],
        input_data['rest_type'],
        input_data['location'],
        input_data['approx_cost(for two people)'],
        input_data['listed_in(type)'],
        input_data['listed_in(city)']
    )
    
    st.subheader('Prediction')
    st.write('Success' if result == 1 else 'Failure')
