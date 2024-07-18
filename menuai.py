from dotenv import load_dotenv
import os
import json
from openai import OpenAI
from embedchain import App
import pandas as pd
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

assistant = client.beta.assistants.create(
  name="Restaurant Server Assistant",
  instructions="You are professional server, you know the menu to about every restaurant and are able to understand fancy names to menu items for people who may not know what is the item about",
  model="gpt-3.5-turbo",
  tools=[{"type": "file_search"}],
)

popular_dishes = {
    'Italian': ['Pizza', 'Pasta', 'Risotto', 'Gelato', 'Tiramisu', 'Lasagna', 'Carbonara', 'Bruschetta', 'Caprese Salad', 'Focaccia', 'Minestrone', 'Pesto', 'Ravioli', 'Gnocchi', 'Osso Buco', 'Arancini', 'Polenta', 'Panettone', 'Cannoli', 'Chianti'],
    'Chinese': ['Dim Sum', 'Peking Duck', 'Sweet and Sour Pork', 'Kung Pao Chicken', 'Fried Rice', 'Spring Rolls', 'Mapo Tofu', 'Wonton Soup', 'Chow Mein', 'Hot Pot', 'Dumplings', 'Char Siu', 'Congee', 'Egg Foo Young', 'Mooncake', 'Baozi', 'Sichuan Hotpot', 'Xiao Long Bao', 'Buddha’s Delight', 'Tea Eggs'],
    'Mexican': ['Tacos', 'Burritos', 'Enchiladas', 'Guacamole', 'Churros', 'Quesadillas', 'Tamales', 'Pozole', 'Fajitas', 'Sopes', 'Mole', 'Elote', 'Chile Relleno', 'Ceviche', 'Carnitas', 'Tostadas', 'Huevos Rancheros', 'Flan', 'Pico de Gallo', 'Agua Fresca'],
    'Indian': ['Curry', 'Biryani', 'Naan', 'Samosa', 'Masala Dosa', 'Tandoori Chicken', 'Rogan Josh', 'Palak Paneer', 'Chole Bhature', 'Butter Chicken', 'Aloo Gobi', 'Dal Tadka', 'Pani Puri', 'Gulab Jamun', 'Jalebi', 'Pav Bhaji', 'Vindaloo', 'Mutter Paneer', 'Kheer', 'Lassi'],
    'Japanese': ['Sushi', 'Ramen', 'Tempura', 'Sashimi', 'Teriyaki', 'Udon', 'Miso Soup', 'Takoyaki', 'Okonomiyaki', 'Onigiri', 'Katsu', 'Gyoza', 'Mochi', 'Sukiyaki', 'Tonkatsu', 'Yakitori', 'Shabu-Shabu', 'Unagi', 'Natto', 'Taiyaki'],
    'French': ['Croissant', 'Baguette', 'Coq au Vin', 'Ratatouille', 'Crème Brûlée', 'Bouillabaisse', 'Quiche', 'Escargot', 'Soufflé', 'Beef Bourguignon', 'Crepes', 'Cassoulet', 'Tarte Tatin', 'Moules Frites', 'Pâté', 'Rillettes', 'Foie Gras', 'Duck Confit', 'Macarons', 'Canelé'],
    'Thai': ['Pad Thai', 'Tom Yum Goong', 'Green Curry', 'Som Tum', 'Mango Sticky Rice', 'Massaman Curry', 'Pad See Ew', 'Khao Soi', 'Larb', 'Tom Kha Gai', 'Satay', 'Red Curry', 'Thai Fish Cakes', 'Boat Noodles', 'Panang Curry', 'Jasmine Rice', 'Khanom Krok', 'Thai Iced Tea', 'Miang Kham', 'Nam Tok'],
    'Greek': ['Gyro', 'Moussaka', 'Souvlaki', 'Greek Salad', 'Baklava', 'Spanakopita', 'Tzatziki', 'Dolmades', 'Kleftiko', 'Avgolemono Soup', 'Feta Cheese', 'Pita Bread', 'Kalamari', 'Saganaki', 'Pastitsio', 'Gigantes Plaki', 'Loukoumades', 'Horiatiki', 'Koulouri', 'Kataifi'],
    'Spanish': ['Paella', 'Tapas', 'Churros', 'Gazpacho', 'Jamón Ibérico', 'Tortilla Española', 'Patatas Bravas', 'Pimientos de Padrón', 'Croquetas', 'Calamares', 'Bocadillo', 'Sangria', 'Empanadas', 'Fabada', 'Crema Catalana', 'Pulpo a la Gallega', 'Albóndigas', 'Salmorejo', 'Manchego Cheese', 'Turrón'],
    'American': ['Hamburger', 'Hot Dog', 'BBQ Ribs', 'Apple Pie', 'Fried Chicken', 'Mac and Cheese', 'Buffalo Wings', 'Clam Chowder', 'Cornbread', 'Meatloaf', 'Jambalaya', 'Gumbo', 'Pecan Pie', 'Cheeseburger', 'New York Pizza', 'Lobster Roll', 'Philly Cheesesteak', 'Biscuits and Gravy', 'Pancakes', 'Chicken Pot Pie'],
    'Turkish': ['Kebab', 'Meze', 'Baklava', 'Döner', 'Turkish Delight', 'Lahmacun', 'Börek', 'Pide', 'Simit', 'Manti', 'Imam Bayildi', 'Iskender Kebab', 'Kumpir', 'Menemen', 'Mercimek Soup', 'Sucuk', 'Ezogelin Soup', 'Köfte', 'Gözleme', 'Kunefe'],
    'Lebanese': ['Hummus', 'Falafel', 'Shawarma', 'Tabouleh', 'Baklava', 'Kibbeh', 'Fattoush', 'Manakish', 'Baba Ghanoush', 'Kafta', 'Mujadara', 'Labneh', 'Arayes', 'Sfeeha', 'Knafeh', 'Warak Enab', 'Makdous', 'Foul Mudammas', 'Shanklish', 'Lamb Chops'],
    'Vietnamese': ['Pho', 'Banh Mi', 'Spring Rolls', 'Bun Cha', 'Cao Lau', 'Com Tam', 'Goi Cuon', 'Nem Ran', 'Banh Xeo', 'Banh Cuon', 'Bo Kho', 'Canh Chua', 'Hu Tieu', 'Mi Quang', 'Xoi', 'Che', 'Ca Phe Trung', 'Goi Du Du', 'Thit Kho To', 'Cha Ca'],
    'Korean': ['Kimchi', 'Bibimbap', 'Bulgogi', 'Korean BBQ', 'Tteokbokki', 'Japchae', 'Samgyeopsal', 'Sundubu-jjigae', 'Galbi', 'Kimchi Jjigae', 'Haemul Pajeon', 'Hoddeok', 'Gimbap', 'Jjajangmyeon', 'Budae Jjigae', 'Dak Galbi', 'Naengmyeon', 'Hanjeongsik', 'Soondae', 'Patbingsu'],
    'Brazilian': ['Feijoada', 'Churrasco', 'Brigadeiro', 'Pão de Queijo', 'Acarajé', 'Moqueca', 'Coxinha', 'Pastel', 'Vatapá', 'Farofa', 'Picanha', 'Tapioca', 'Empadão', 'Feijão Tropeiro', 'Quindim', 'Canjica', 'Pudim', 'Bolinhos de Bacalhau', 'Mandioca Frita', 'Caipirinha'],
    'Ethiopian': ['Injera', 'Doro Wat', 'Tibs', 'Kitfo', 'Sambusa', 'Shiro', 'Misir Wot', 'Kik Alicha', 'Berbere', 'Atayef', 'Firfir', 'Genfo', 'Chechebsa', 'Tihlo', 'Gomen', 'Enkulal Firfir', 'Azifa', 'Yetsom Beyaynetu', 'Tegabino Shiro', 'Awaze'],
    'Moroccan': ['Tagine', 'Couscous', 'Harira', 'Pastilla', 'Moroccan Mint Tea', 'Zaalouk', 'Rfissa', 'Briouats', 'Chebakia', 'Baghrir', 'Harsha', 'Mechoui', 'Maakouda', 'Seffa', 'Khobz', 'Mrouzia', 'Batbout', 'Ghoriba', 'Sellou', 'Tanjia'],
    'Peruvian': ['Ceviche', 'Lomo Saltado', 'Aji de Gallina', 'Anticuchos', 'Pisco Sour', 'Rocoto Relleno', 'Papa a la Huancaína', 'Arroz con Pollo', 'Causa Rellena', 'Tacu Tacu', 'Leche de Tigre', 'Pollo a la Brasa', 'Chicha Morada', 'Chupe de Camarones', 'Carapulcra', 'Lucuma', 'Picarones', 'Solterito', 'Adobo', 'Quinoa'],
    'Indonesian': ['Nasi Goreng', 'Satay', 'Rendang', 'Gado-Gado', 'Bakso', 'Soto', 'Martabak', 'Gudeg', 'Ayam Goreng', 'Pempek', 'Lumpia']
}


txt_path = "popular_dishes.txt"


text_file = client.files.create(
    file=open(txt_path, "rb"), purpose="assistants"
)


# Upload the user-provided PDF file to OpenAI
pdf_file = client.files.create(
    file=open("menu_items.pdf", "rb"), purpose="assistants"
)


vector_store = client.beta.vector_stores.create(name='menu_items')
print(vector_store)

file_paths = ['./menu_items.pdf']
file_streams = [open(file_path, 'rb') for file_path in file_paths]

file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id,
    files=file_streams,
)
print(file_batch.status)
print(file_batch.file_counts)
print(file_batch)


assistant = client.beta.assistants.update(
    assistant_id=assistant.id,
    tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
)


# Create a thread and attach the file to the message
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "Use the items on the menu as an example of possible exotic dishes",
            "attachments": [
                { "file_id": text_file.id, "tools": [{"type": "file_search"}] },
                { "file_id": pdf_file.id, "tools": [{"type": "file_search"}] }
            ],
        }
    ]
)


run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id
)

messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

print(messages)
message_content = messages[0].content[0].text
annotations = message_content.annotations
citations = []
for index, annotation in enumerate(annotations):
    message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    if file_citation := getattr(annotation, "file_citation", None):
        cited_file = client.files.retrieve(file_citation.file_id)
        citations.append(f"[{index}] {cited_file.filename}")

print(message_content.value)
print("\n".join(citations))


message = client.beta.threads.messages.create(
    thread_id = thread.id,
    role = 'user',
    content = 'can you describe what a pastelito is?'
)


# Use the create and poll SDK helper to create a run and poll the status of
# the run until it's in a terminal state.

run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id, assistant_id=assistant.id
)

messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

message_content = messages[0].content[0].text
annotations = message_content.annotations
citations = []
for index, annotation in enumerate(annotations):
    message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
    if file_citation := getattr(annotation, "file_citation", None):
        cited_file = client.files.retrieve(file_citation.file_id)
        citations.append(f"[{index}] {cited_file.filename}")

print(message_content.value)
print("\n".join(citations))