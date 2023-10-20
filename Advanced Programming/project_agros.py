"""
Project Agros is set to study the agricultural output of several countries.

Class:
    Agros: Is set to help analyze a dataset that
    explores the Agriculture aspects of many countries.
    It contains eight methods to help analyze the
    outputs based on the inputs provided by the dataset.
"""
import os
import io
import zipfile
import warnings
import requests
import pycountry
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Ignore the warnings, as some parameters may be incompatible with some ARIMA settings
warnings.filterwarnings("ignore")


class Agros:

    """
    The Agros class is used to study the agricultural output of several countries.

    Variables:
        merge_dict:
            A dictionary that renames countries.

    Attributes:
        url_agro (str):
            A link to get the agricultural data.
        name_agro (str):
            The name to be given to the dataset.
        url_geo (str):
            A link to get the geospatial data.
        name_geo (str):
            The name to be given to the dataset.

    Methods:
        load_data():
        Downloads the data files and reads it into DataFrames.

        countries_list() -> list:
        It takes the dataset into a count and returns a list of all the unique countries.

        correlation():
        This method creates a new data frame that with the quantities
        and computes it into a correlation matrix.

        plot_area_chart(country=None, normalize=False):
        This method takes two arguments and computes an area chart of the crop, animals and fish
        outputs of a certain country over the period of time present in the dataset.

        display_output(countries):
        This method plots a linear chart of the outputs, over the years,
        based on a provided country/countries.

        gapminder(year, x_log_scale:False, y_log_scale:False):
        This method checks if the argument given is truly an year,
        present in the dataset, and computes a scatter plot of the output_quantity
        by the fertalizer_quantity, of all the countries in that year.

        chropleth(year):
        This method checks if the argument given is truly an year,
        merges our datasets in order to plot a chropleth,
        of the Total Factory Productivity of the giver year,
        this allows us to compare the tfp of each country.

        predictor(countries):
        This method takes a list up to three countries and
        plots the total factor productivity, by year and
        complements it with an ARIMA prediction up to the year 2050.

    """

    merge_dict = {
        "United States of America": "United States",
        "Greenland": "Greenland",
        "Dominican Rep.": "Dominican Republic",
        "Republic of Serbia": "Serbia",
        "United Republic of Tanzania": "Tanzania",
        "Central African Rep.": "Central African Republic",
        "South Sudan": "Sudan",
        "Democratic Republic of the Congo": "Democratic Republic of Congo",
        "Republic of the Congo": "Congo",
        "Somaliland": "Somalia",
        "eSwatini": "Eswatini",
        "Antarctica": "Antartica",
        "Bosina and Herz.": "Bosnia and Herzegovina",
        "Solomon Is.": "Solomon Islands",
        "Timor-Leste": "Timor",
    }

    def __init__(
        self,
        url_agro: str = "https://github.com/owid/owid-datasets"
        + "/blob/master/datasets/"
        + "Agricultural%20total%20factor%20productivity"
        + "%20(USDA)/Agricultural%20total"
        + "%20factor%20productivity%20(USDA).csv",
        name_agro: str = "Agricultural_analysis.csv",
        url_geo: str = "https://www.naturalearthdata.com/"
        + "http//www.naturalearthdata.com/download/110m/"
        + "cultural/ne_110m_admin_0_countries.zip",
        name_geo: str = "ne_110m_admin_0_countries.shp",
    ):
        self.url_agro = url_agro
        self.name_agro = name_agro
        self.url_geo = url_geo
        self.name_geo = name_geo

    def load_data(self):
        """
        Ensure:
        ---------------
        This method will download the data file into
        a downloads/ directory in the root directory of the project.
        If the data file already exists, the method will not download it again.

        Returns:
        ---------------
        self.data:
            An attribute that allows us to call the Agricultural dataset
            and present in pd.Dataframe() format.
        self.data_geo:
            An attribute that allows us to call the geospatial dataset
            and present it as a DataFrame.

        """

        filename_agro = self.name_agro
        fileurl_agro = self.url_agro
        filename_geo = self.name_geo
        fileurl_geo = self.url_geo

        # set up download directory
        download_dir = "downloads"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        # check if file already exists in download directory
        file_path_agro = os.path.join(download_dir, filename_agro)
        file_path_geo = os.path.join(download_dir, filename_geo)
        if os.path.exists(file_path_agro) and os.path.exists(file_path_geo):
            print(
                f"Data file '{filename_agro}' already exists in '{download_dir}' directory.\n"
                f"Data file '{filename_geo}' already exists in '{download_dir}' directory."
            )
            self.data = pd.read_csv(file_path_agro)  # pylint: disable=W0201
            data_geo = gpd.read_file(file_path_geo)
        else:
            # download file (it's already a csv)
            file = pd.read_csv(fileurl_agro + "?raw=true")

            # Get file from the website send a request with a custom user-agent header
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                + " (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
            }
            request = requests.get(fileurl_geo, headers=headers)

            # save file to download directory
            zip_file = zipfile.ZipFile(io.BytesIO(request.content))
            zip_file.extractall("downloads")
            print(
                f"Data file '{filename_geo}' downloaded and saved to '{download_dir}' directory."
            )
            file.to_csv(file_path_agro, index=False)
            print(
                f"Data file '{filename_agro}' downloaded and saved to '{download_dir}' directory."
            )
            self.data = file  # pylint: disable=W0201
            data_geo = gpd.read_file(file_path_geo)
        data_geo = data_geo[["ADMIN", "geometry"]]

        # read as DataFrames
        self.path = os.path.join("downloads", self.name_agro)  # pylint: disable=W0201
        self.path_geo = os.path.join("downloads", self.name_geo)
        self.data = pd.read_csv(self.path)  # pylint: disable=W0201
        self.data_geo = data_geo

        # creating list with countries in Entity column, using is_country function
        countries_list = []
        entities_list = list(self.data["Entity"].unique())
        for country in entities_list:
            try:
                pycountry.countries.search_fuzzy(country)
                countries_list.append(country)
            except LookupError:
                pass

        # Add countries which were removed incorrectly and remove not detected regions
        missing_countries = [
            "Cape Verde",
            "Czechoslovakia",
            "Democratic Republic of Congo",
        ]
        not_detected_regions = ["Central Africa"]
        countries_list = sorted(
            list(
                filter(
                    lambda x: x not in not_detected_regions,
                    countries_list + missing_countries,
                )
            )
        )

        self.data = self.data[self.data["Entity"].isin(countries_list)]

    def countries_list(self) -> list:
        """
        Ensure:
        ---------------
        List of all the countries available.

        Returns:
        ---------------
        lista:
          Unique values (countries) of the "Entity" column.

        """
        data = self.data
        lista = list(data["Entity"].unique())
        return lista

    def correlation(self):
        """
        Ensure:
        ---------------
        A dataframe with all the columns of the previus
        dataframe that have the following sequencia of caracteres
        "_quantity", and computes a correlation matrix with the data
        of the dataframe and in the sametime plot a heat map.

        Returns:
        ---------------
        plt.show()
            A correlation matrix between all the quantity columns on the dataset

        """
        df_quantity = self.data.filter(regex="_quantity")
        col_corr = df_quantity.corr()
        mask = np.triu(np.ones_like(col_corr, dtype=bool))

        fig, axis = plt.subplots(figsize=(10, 8))
        sns.heatmap(col_corr, mask=mask, cmap="GnBu", annot=True, ax=axis)
        axis.set_title("Correlation Matrix of Agricultural Production Outputs")
        axis.text(
            -0.2,
            -0.3,
            "Source: Agricultural Total Factor Productivity (USDA)",
            transform=axis.transAxes,
            fontsize=10,
            ha="left",
        )

        return plt.show()

    def plot_area_chart(self, country=None, normalize=False):

        """
        Ensure:
        ---------------
        A plot graph area with the variables crop_out_quantity, animal_out_quantity and
        fish_output_quantity of a certain country over the period of time present in the dataset.

        Parameters:
        ---------------
        country(str) = None:
            A country should be a string, if 'World' or None it will output the
            chart for every country.

        normalize(boll):
            It has default value as False, if True will normalize the output.

        Raises:
        ---------------
            ValueError: If country is not in the dataset

        Returns:
        ---------------
            A area of the different kinds of outputs, of a countries
            agricultural production.

        """

        data = self.data
        # Get columns containing '_output_'
        output_cols = [col for col in data.columns if "_output_" in col]

        # If country is None or 'World', sum output for all countries
        if country is None or country == "World":
            grouped_data = data.groupby(["Year"])[
                output_cols
            ].sum()  # pylint: disable=E1101
        else:
            # Check if chosen country exists in the data
            if country not in data["Entity"].unique():
                raise ValueError("Chosen country does not exist in the data")
            # Filter data by chosen country
            filtered_data = data[data["Entity"] == country]
            grouped_data = filtered_data.groupby(["Year"])[output_cols].sum()

        # If normalize is True, normalize output in relative terms
        if normalize:
            grouped_data = grouped_data.div(grouped_data.sum(axis=1), axis=0) * 100
            grouped_data_plot = grouped_data.plot(kind="area", stacked=True, alpha=0.5)
            grouped_data_plot.set_ylabel("Percentage")
        else:
            grouped_data_plot = grouped_data.plot(kind="area", stacked=False, alpha=0.5)
            grouped_data_plot.set_ylabel("Sum")
        # Set labels and title
        grouped_data_plot.set_xlabel("Year")
        grouped_data_plot.set_title(
            f"Area Chart of Distinct Columns Containing '_output_' from {country}",
            y=1.05,
        )
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.text(
            0,
            -0.1,
            "Source: Agricultural Total Factor Productivity (USDA) ",
            transform=plt.gcf().transFigure,
            fontsize=8,
        )
        # Show plot
        plt.show()

    def display_output(self, countries):
        """
        Ensure:
        ---------------
        This method plots a linear chart of the outputs, over the years,
        based on a provided country/countries.

        Parameters:
        ---------------
        countries(str/list):
            Countries should be either a string or list format.

        Returns:
        ---------------
            Graph chart with output of countries by year

        """

        # If a string is provided, convert it to a list with that country

        data = self.data
        if isinstance(countries, str):
            c_list = [countries]
        else:
            c_list = countries

        for country_x in c_list:

            # Define x and y for each country (line in graph)

            x_var = list(data[data["Entity"] == country_x]["Year"])
            y_var = list(data[data["Entity"] == country_x]["output"])

            # Plot the line chart

            plt.plot(x_var, y_var, label=country_x)

        # Add labels, title and legend

        plt.xlabel("Year")
        plt.ylabel("Output")
        plt.title("Output of countries by year")
        plt.legend(loc="upper left", fontsize="medium", bbox_to_anchor=(1, 1))
        plt.text(
            0,
            -0.1,
            "Source: Agricultural Total Factor Productivity (USDA) ",
            transform=plt.gcf().transFigure,
            fontsize=8,
        )
        # Show the chart

        plt.show()

    def gapminder(self, year, x_log_scale: False, y_log_scale: False):
        """
        Ensure:
        ---------------
        This method checks if the argument given is truly an year,
        present in the dataset, and computes a scatter plot of the output_quantity
        by the fertalizer_quantity, of all the countries in that year.

        Parameters:
        ---------------
        year(int):
            The year should be an integer.
        x_log_scale(boll):
            If True x axis will type 'log'.
        y_log_scale(boll):
            If True y axis will type 'log'.

        Raises:
        ---------------
            TypeError: If year not and integer
            ValueError: If year is not contained in dataset

        Returns:
        ---------------
            Scatter plot of output_quantity by fertilizer_quantity
            with size equal to machinery_quantity, of the respecive year.
        """
        data = self.data
        if isinstance(year, (int, np.int, np.int64, np.int32)):
            resp = year
        elif isinstance(year, str) and year.isdigit():
            resp = int(year)
        else:
            raise TypeError("Year must be an integer.")
        if isinstance(resp, (int, np.int, np.int64, np.int32)):
            if (
                resp > data.Year.max() or resp < data.Year.min()
            ):  # pylint: disable=E1101
                raise ValueError(
                    "The year you want it's not in our database,"
                    + f"please select a year from {data.Year.min()} to {data.Year.max()}."  # pylint: disable=E1101
                )

        data_year = data[data["Year"] == resp]
        x_var = data_year["fertilizer_quantity"]
        y_var = data_year["output_quantity"]
        a_var = data_year["machinery_quantity"]

        plt.figure(dpi=120)
        sns.scatterplot(x=x_var, y=y_var, size=a_var, sizes=(20, 600))

        plt.grid(True)

        plt.xlabel("Fertilizer Quantity")
        plt.ylabel("Output Quantity")

        # Move the circle size legend outside of the plot area
        plt.legend(title="Machinery Quantity", loc="upper left", bbox_to_anchor=(1, 1))
        plt.title(f"Gapminder({year})")
        if x_log_scale is True:
            plt.xscale("log")
        if y_log_scale is True:
            plt.yscale("log")
        plt.text(
            0,
            -0.1,
            "Source: Agricultural Total Factor Productivity (USDA) ",
            transform=plt.gcf().transFigure,
            fontsize=8,
        )
        plt.show()

    def choropleth(self, year: int):

        """
        Ensure:
        ---------------
        This method checks if the argument given is truly an year,
        present in the dataset, and computes a choropleth of the total factory productivity,
        of all the countries in that year.

        Parameters:
        ---------------
        year(int):
            The year should be an integer.

        Raises:
        ---------------
            TypeError: If year not and integer..
            ValueError: If year is not contained in dataset

        Returns:
        ---------------
        map0
            Choropleth of TFP(total factory productivity) of each country, of the respecive year.
        """

        if not isinstance(year, (int, np.int, np.int64, np.int32)):
            raise TypeError("Year must be an integer.")
        data_agro = self.data
        data_geo = self.data_geo
        data_geo["ADMIN"].replace(self.merge_dict, inplace=True)

        data = data_geo.merge(data_agro, how="left", left_on="ADMIN", right_on="Entity")

        if isinstance(year, (int, np.int, np.int64, np.int32)):
            resp = year
        elif isinstance(year, str) and year.isdigit():
            resp = int(year)
        else:
            raise TypeError("Year must be an integer.")
        if isinstance(resp, (int, np.int, np.int64, np.int32)):
            if (
                resp > data.Year.max() or resp < data.Year.min()
            ):  # pylint: disable=E1101
                raise ValueError(
                    "The year you want it's not in our database,"
                    + f"please select a year from {int(data.Year.min())} to {int(data.Year.max())}."  # pylint: disable=E1101
                )
        # Choropleth
        map0 = folium.Map(location=(20, 10), zoom_start=1.5, tiles="cartodb positron")

        # data and design
        choropleth_layer = folium.Choropleth(
            geo_data=data[data["Year"] == resp],
            data=data[data["Year"] == resp],
            columns=["ADMIN", "tfp"],
            key_on="feature.properties.ADMIN",
            fill_color="YlGnBu",
            fill_opacity=0.8,
            line_opacity=0.3,
            nan_fill_color="white",
            legend_name=f"Total Factor Productivity({year})",
        ).add_to(map0)

        # add tooltip
        tooltip = folium.GeoJsonTooltip(
            fields=["ADMIN", "tfp"],
            aliases=["Country:", f"TFP ({year}):"],
            localize=True,
            sticky=True,
            labels=True,
            opacity=0.7,
            direction="top",
            style="""
                font-size: 12px;
                color: #333333;
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 4px;
             """,
        )

        choropleth_layer.geojson.add_child(tooltip)

        # Add title and source
        title_html = (
            "<h3 align='center'' style='font-size:16px'>"
            f"<b>Total Factor Productivity ({year})</b></h3>"
            "<p align='center' style='font-size:12px'>"
            "Source: Natural Earth (naturalearthdata.com) & "
            "Agricultural Total Factor Productivity (USDA)</p>"
        )
        map0.get_root().html.add_child(folium.Element(title_html))

        return map0

    def predictor(self, countries):
        """
        Ensure:
        ---------------
        The method takes a list up to three countries and plots the total factor productivity,
        by year and complements it with an auto-ARIMA prediction up to the year 2050.

        Parameters:
        ---------------
        countries(str/list):
            Countries should be either a string or list format.

        Raises:
        ---------------
            ValueError: If countries not in the dataset.

        Returns:
        ---------------
        plt.show():
            A time series, with a prediction of tfp until 2050.

        """
        # Filter dataframe by country names in input list
        df_countries = self.data[self.data["Entity"].isin(countries)]

        # Check if any of the input countries are missing from the dataframe
        missing_countries = set(countries) - set(df_countries["Entity"])
        if missing_countries:
            missing_countries_str = ", ".join(missing_countries)
            raise ValueError(
                "The following countries were not found in the "
                + f"Agricultural dataframe and will be ignored: {missing_countries_str}"
            )

        # Plot actual TFP data for each country
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap("Set1", len(countries))
        for i, country in enumerate(countries):
            df_country = df_countries[df_countries["Entity"] == country]
            df_country["Year"] = pd.to_datetime(df_country["Year"], format="%Y")
            df_country["Year"] = df_country["Year"].dt.year
            plt.plot(
                df_country["Year"],
                df_country["tfp"],
                color=colors(i),
                linestyle="-",
                label=f"{country} Actual",
            )

            # Fit ARIMA model and plot predicted TFP data
            stepwise_fit = auto_arima(
                df_country["tfp"],
                start_p=1,
                start_q=1,
                max_p=5,
                max_q=5,
                m=1,
                start_P=0,
                seasonal=True,
                d=None,
                D=0,
                trace=False,
                error_action="ignore",  # Ignore incompatible settings
                suppress_warnings=True,
                stepwise=True,
            )

            prediction = pd.DataFrame(stepwise_fit.predict(n_periods=31))
            prediction["Year"] = pd.date_range(start="2019", periods=31, freq="YS")
            prediction["Year"] = prediction["Year"].dt.year
            prediction.set_index("Year", inplace=True)
            plt.plot(
                prediction.index,
                prediction.values,
                color=colors(i),
                linestyle="--",
                label=f"{country} Predicted",
            )

        # Add labels, title, and legend to the plot
        plt.legend()
        plt.xlabel("Year")
        plt.ylabel("TFP")
        plt.title("Actual and predicted TFP for selected countries")

        # Set x-axis range
        plt.xlim([1960, 2050])
        plt.text(
            0,
            -0.1,
            "Source: Agricultural Total Factor Productivity (USDA) ",
            transform=plt.gcf().transFigure,
            fontsize=8,
        )
        # Show plot
        return plt.show()
