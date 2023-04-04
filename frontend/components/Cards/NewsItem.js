import React from 'react';
import Image from "next/image";


const NewsItem = ({newTitle, newsImgUrl, newsUrl, newsDesc}) => {
  return (
    <div class="w-full lg:w-1/3 rounded overflow-hidden shadow-lg m-2 relative flex flex-col ">
        <img
          src={newsImgUrl}
          className="newsItemImg"
          alt="Demeter Logo"
        />
        <div class="px-4 py-2">
          <div class="text-base font-semibold tracking-tight leading-none mb-1 newsItemTitle">{newTitle}</div>
          <p class="text-sm tracking-tighter leading-tight newsItemDesc">
            {newsDesc}
          </p>
        </div>
        <a class="w-full bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-1 px-4 rounded inline-flex items-center" href={newsUrl} rel="noreferrer" target="_blank">
          <span>Read More...</span>
        </a>
      </div>
  )
}

export default NewsItem;