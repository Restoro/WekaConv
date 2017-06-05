package Enums;

public enum Classes {
	beach, 
	bus, 
	cafe_restaurant, 
	car, city_center, 
	forest_path,
	grocery_store, 
	home, 
	library, 
	metro_station,
	office, park,
	residential_area,
	train, 
	tram;
	
	private static Classes[] allValues = values();
    public static Classes fromOrdinal(int n) {return allValues[n];}
}
