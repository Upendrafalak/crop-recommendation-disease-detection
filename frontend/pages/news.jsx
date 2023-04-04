import React, { useState, useEffect } from "react";
import { signIn, useSession, getSession } from "next-auth/react";
import NewsItem from "../components/Cards/NewsItem.js";
import AdminNavbar from "../components/Navbars/AdminNavbar.js";
import Sidebar from "../components/Sidebar/Sidebar.js";
import FooterAdmin from "../components/Footers/FooterAdmin.js";
import Layout from "../components/Layout";
import axios from "axios";

export default function News() {
  const { data: session, status } = useSession();
  console.log(session);
  const [loading, setLoading] = useState(true);
  const [totalNews, setTotalNews] = useState([]);
  const [count, setCount] = useState(0);

  useEffect(() => {
    const securePage = () => {
      if (status === "unauthenticated") {
        signIn();
      } else {
        setLoading(false);
      }
    };
    securePage();
    axios
      .get(`${process.env.NEXT_PUBLIC_BACKEND_URL}/getnews`)
      .then(function (response) {
        setTotalNews(response.data.articles);
        console.log(response.data);
      })
      .catch(function (error) {
        console.log(error);
      });
  }, []);

  if (loading) {
    return <h2 style={{ marginTop: 100, textAlign: "center" }}>LOADING...</h2>;
  }
  return (
    <Layout title="News">
      <Sidebar />
      <div className="relative md:ml-64 bg-blueGray-100">
        <AdminNavbar title={"News"} image={session.user.image} />
        <div className="relative bg-blueGray-800 md:pt-32 pb-6 pt-12"></div>
        <div className="px-4 md:px-10 mx-auto w-full -m-24">
          <div className="lg:flex mb-4 m-4 rounded mx-auto">
            {totalNews.slice(count, count + 3).map((news) => {
              return (
                <NewsItem
                  key={news.publishedAt}
                  newTitle={news.title}
                  newsImgUrl={news.urlToImage}
                  newsUrl={news.url}
                  newsDesc={news.description}
                />
              );
            })}
          </div>
          <div className="lg:flex mb-4 m-4 rounded mx-auto">
            {totalNews.slice(count + 3, count + 6).map((news) => {
              return (
                <NewsItem
                  key={news.publishedAt}
                  newTitle={news.title}
                  newsImgUrl={news.urlToImage}
                  newsUrl={news.url}
                  newsDesc={news.description}
                />
              );
            })}
          </div>
          <div className="flex flex-col items-center">
            <div className="inline-flex mt-2 xs:mt-0">
              <button
                className="px-4 py-2 mx-4 text-sm font-medium text-white bg-gray-800 rounded-l hover:bg-gray-900 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
                onClick={() => {
                  setCount(count - 6);
                  console.log(count);
                }}
              >
                Prev
              </button>
              <button
                className="px-4 py-2 text-sm font-medium text-white bg-gray-800 border-0 border-l border-gray-700 rounded-r hover:bg-gray-900 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white"
                onClick={() => {
                  setCount(count + 6);
                  console.log(count);
                }}
              >
                Next
              </button>
            </div>
          </div>
          <FooterAdmin />
        </div>
      </div>
    </Layout>
  );
}

export async function getServerSideProps(context) {
  const session = await getSession(context);
  let userId = null;

  return {
    props: {
      session,
      userId,
    },
  };
}
