import streamlit as st
import pandas as pd
import google.generativeai as genai
from PIL import Image
import os
import json

# --- Configuration ---
# Configure the Gemini API key.
# IMPORTANT: For public hosting, use Streamlit's secrets management.
# For local testing, you can set it as an environment variable.
try:
    # This is for Streamlit Community Cloud deployment
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (FileNotFoundError, KeyError):
    # This is for local development
    # Make sure you have a .streamlit/secrets.toml file with your key
    st.warning("Could not find Streamlit secrets. Make sure you have a .streamlit/secrets.toml file for local testing.")
    # As a fallback for simple local run, you might use an environment variable, but secrets are better.
    # genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


# --- Gemini Pro Vision Model ---
def get_gemini_response(image, prompt):
    """
    Analyzes the receipt image using the Gemini Pro Vision model.
    """
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([image, prompt])
    return response.text


# --- Main Application ---

# Page Configuration
st.set_page_config(
    page_title="AI Receipt Processor",
    page_icon="🧾",
    layout="wide"
)

# App Title
st.title("🧾 AI-Powered Receipt Processor")
st.write(
    "For long receipts, you can upload multiple pictures. The AI will combine the items from all images and remove any duplicates.")

# --- File Uploader and Processing ---
# Allow multiple files to be uploaded
uploaded_files = st.file_uploader("Choose receipt images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # --- NEW: Save uploaded images to a local folder ---
    IMAGE_SAVE_FOLDER = 'uploaded_receipts'
    if not os.path.exists(IMAGE_SAVE_FOLDER):
        os.makedirs(IMAGE_SAVE_FOLDER)

    st.subheader("Uploaded Receipt Images")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        # Display the image in the app
        st.image(image, caption=uploaded_file.name, width=300)

        # Save the image to the specified folder
        save_path = os.path.join(IMAGE_SAVE_FOLDER, uploaded_file.name)
        image.save(save_path)

    st.success(f"All {len(uploaded_files)} image(s) have been saved to the '{IMAGE_SAVE_FOLDER}' folder.")
    # --- End of new image saving section ---

    if st.button("Process All Receipts", type="primary"):
        with st.spinner("🧠 The AI is reading your receipt(s)... This may take a moment."):

            # Initialize lists to store combined results
            all_line_items = []
            final_date = "N/A"
            final_total = 0.0

            # The prompt is updated to handle partial receipts
            prompt = """
            You are an expert in processing grocery store receipts.
            Analyze the following receipt image. It might be one part of a longer receipt.
            Extract these details if they are present:
            1.  **Purchase Date**: The date of the transaction in YYYY-MM-DD format. If not present, return null.
            2.  **Total Amount**: The final total amount paid. If not present, return null.
            3.  **Line Items**: A list of all purchased items visible in this image. For each item, extract its description, quantity (if available, otherwise 1), and price.

            Return the information as a clean JSON object. Do not include any text before or after the JSON.
            The JSON structure should be:
            {
              "purchase_date": "YYYY-MM-DD" or null,
              "total_amount": 0.00 or null,
              "line_items": [
                {
                  "description": "Item Name",
                  "quantity": 1,
                  "price": 0.00
                }
              ]
            }
            """
            try:
                # Loop through each uploaded file again for processing
                for uploaded_file in uploaded_files:
                    st.write(f"Processing `{uploaded_file.name}`...")
                    image = Image.open(uploaded_file)

                    # Call the Gemini API for each image
                    response_text = get_gemini_response(image, prompt)

                    # Clean the response to ensure it's valid JSON
                    cleaned_response = response_text.strip().replace("```json", "").replace("```", "")
                    receipt_data = json.loads(cleaned_response)

                    # Aggregate the data
                    if receipt_data.get("line_items"):
                        all_line_items.extend(receipt_data["line_items"])

                    # Update date and total if found (last one wins, assuming it's at the end)
                    if receipt_data.get("purchase_date"):
                        final_date = receipt_data["purchase_date"]
                    if receipt_data.get("total_amount"):
                        final_total = receipt_data["total_amount"]

                st.success("All receipt images processed successfully!")

                # --- Deduplication Step ---
                if all_line_items:
                    st.info("Removing duplicate items from overlapping images...")
                    raw_items_df = pd.DataFrame(all_line_items)

                    # Ensure price and description columns exist before dropping duplicates
                    if 'description' in raw_items_df.columns and 'price' in raw_items_df.columns:
                        # Drop duplicates based on both description and price
                        items_df = raw_items_df.drop_duplicates(subset=['description', 'price'], keep='first')
                    else:
                        # Fallback if columns are missing, though unlikely with the prompt
                        items_df = raw_items_df
                else:
                    items_df = pd.DataFrame()  # Create empty dataframe if no items found

                # --- Display Combined Extracted Data ---
                st.subheader("Consolidated Information")
                col1, col2 = st.columns(2)
                col1.metric("Purchase Date", final_date)
                col2.metric("Total Amount", f"${final_total:.2f}")

                st.write("All Purchased Items (Duplicates Removed):")
                st.dataframe(items_df)

                # --- Save to CSV ---
                if not items_df.empty:
                    st.subheader("Save Data to CSV")

                    # Prepare data for CSV
                    csv_df = items_df.copy()
                    csv_df['purchase_date'] = final_date
                    csv_df['receipt_total'] = final_total

                    csv_file_path = 'receipts_data.csv'

                    # Append to CSV if it exists, otherwise create it
                    if os.path.exists(csv_file_path):
                        existing_df = pd.read_csv(csv_file_path)
                        combined_df = pd.concat([existing_df, csv_df], ignore_index=True)
                    else:
                        combined_df = csv_df

                    combined_df.to_csv(csv_file_path, index=False)
                    st.info(f"Data has been saved to `{csv_file_path}` on the server.")

                    # Provide a download button for the user
                    st.download_button(
                        label="Download All Receipts as CSV",
                        data=combined_df.to_csv(index=False).encode('utf-8'),
                        file_name='all_receipts.csv',
                        mime='text/csv',
                    )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("The AI could not process one of the images. Please ensure all pictures are clear.")

