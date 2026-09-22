import pystac_client
import planetary_computer

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

bbox = [31.30, 36.90, 31.55, 37.10]

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2021-05-01/2021-10-31",
    query={"eo:cloud_cover": {"lt": 15}},
)

items = sorted(search.items(), key=lambda it: it.datetime)

print(f"Toplam {len(items)} görüntü:\n")
for item in items:
    cc = item.properties["eo:cloud_cover"]
    print(f"{item.datetime.date()}  bulutluluk: %{cc:.2f}  id: {item.id}")