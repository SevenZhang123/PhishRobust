# Active domain name detection:
# Based on Censys' query of all domains for the brand,
# a domain list is built.

import tldextract
from censys.search import CensysCerts
import json


api_id = "your-api-id"
api_secret = "your-api-secret"
c = CensysCerts(api_id=api_id, api_secret=api_secret)

f_brand = open("./brand_cert.txt", "r", encoding="utf-8")
for line in f_brand:
    brand_list = []
    brands = line.split("\t")[0]
    Org = line.split("\t")[1]
    if "+" in brands:
        brand_list = brands.split("+")
    else:
        brand_list.append(brands)

    # Target organization name. Must be an exact match.
    target_org = Org
    unique_root_domains = set()

    print(f"Retrieving certificate data for {target_org}...")

    # Query Censys
    query = f'parsed.subject.organization: "{target_org}"'

    try:
        all_results = c.search(query, pages=-1)

        count = 0
        for page_results in all_results:
            for result in page_results:
                count += 1
                if count % 100 == 0:
                    print(f"{count} certificates have been processed")

                if "names" in result:
                    for full_domain in result["names"]:
                        try:
                            # Domain name resolution
                            ext = tldextract.extract(full_domain)

                            if ext.domain and ext.suffix:
                                root_domain = f"{ext.domain}.{ext.suffix}"
                                unique_root_domains.add(root_domain)
                        except Exception:
                            continue

        print(f"{count} certificate records were processed.")

    except Exception as e:
        print(f"An error occurred:{e}")

    print(f"\n=== Query ended ===")
    print(f"A total of {len(unique_root_domains)} unique (registered) domains were discovered:")
    for brand in brand_list:
        brand = brand.replace(" ", "_")
        save_path = "../../data/brand_domain/" + brand + "_domain_list.txt"
        f = open(save_path, "w", encoding="utf-8")
        for domain in sorted(unique_root_domains):
            print(domain)
            f.write(domain + "\n")
        f.close()